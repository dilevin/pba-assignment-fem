# usd_writer.py
from pxr import Usd, UsdGeom, Sdf, Gf, Vt
import numpy as np
from typing import Iterable, Optional, Sequence

def _vec3f(xyz):
    x, y, z = xyz
    return Gf.Vec3f(float(x), float(y), float(z))
    
class USDMeshWriter:
    """
    Stream a deforming triangle mesh to a time-sampled USD file.

    Usage:
        writer = USDMeshWriter("sim.usdc", fps=60, up_axis="Z", write_velocities=True)
        writer.open(face_counts=[3,3,...], face_indices=[...], num_points=N)
        for k in range(num_frames):
            P = get_points_from_sim(k)          # (N,3) np.ndarray / torch / warp
            writer.write_frame(P)               # time = k
        writer.close()
    """

    def __init__(self,
                 output_path: str,
                 fps: float = 24.0,
                 up_axis: str = "Z",
                 meters_per_unit: float = 1.0,
                 prim_path: str = "/DeformingMesh",
                 write_velocities: bool = False,
                 flush_every_frame: bool = True):
        self.output_path = output_path
        self.fps = float(fps)
        self.up_axis = up_axis.upper()
        self.meters_per_unit = float(meters_per_unit)
        self.prim_path = Sdf.Path(prim_path)
        self.write_velocities = write_velocities
        self.flush_every_frame = flush_every_frame

        self._stage: Optional[Usd.Stage] = None
        self._mesh: Optional[UsdGeom.Mesh] = None
        self._points_attr = None
        self._extent_attr = None
        self._vel_attr = None
        self._prev_points: Optional[np.ndarray] = None
        self._frame_index = 0
        self._num_points = None

    def open(self, face_counts, face_indices, num_points: int):
        self._stage = Usd.Stage.CreateNew(self.output_path)
        self._stage.SetTimeCodesPerSecond(self.fps)
        self._stage.SetStartTimeCode(0.0)
        self._stage.SetEndTimeCode(0.0)

        UsdGeom.SetStageUpAxis(
            self._stage,
            UsdGeom.Tokens.z if self.up_axis == "Z" else UsdGeom.Tokens.y
        )
        UsdGeom.SetStageMetersPerUnit(self._stage, self.meters_per_unit)

        self._mesh = UsdGeom.Mesh.Define(self._stage, self.prim_path)

        # --- FIX: coerce to typed Vt arrays
        fc = Vt.IntArray([int(x) for x in list(face_counts)])
        fi = Vt.IntArray([int(x) for x in list(face_indices)])
        self._mesh.GetFaceVertexCountsAttr().Set(fc)
        self._mesh.GetFaceVertexIndicesAttr().Set(fi)

        self._points_attr = self._mesh.GetPointsAttr()
        self._extent_attr = self._mesh.GetExtentAttr()
        self._vel_attr = self._mesh.GetVelocitiesAttr() if self.write_velocities else None

        self._stage.SetDefaultPrim(self._stage.GetPrimAtPath(self.prim_path))
        self._num_points = int(num_points)
        self._frame_index = 0
        self._prev_points = None

        self._stage.GetRootLayer().Save()

    @staticmethod
    def _to_numpy(P: Iterable) -> np.ndarray:
        """
        Accepts:
          - numpy array (N,3)
          - torch.Tensor (N,3)
          - warp.array (N,3)  (cpu: .numpy(); gpu: .to_torch().cpu().numpy())
          - list-like
        Returns float32 numpy array of shape (N,3).
        """
        # PyTorch
        try:
            import torch  # type: ignore
            if isinstance(P, torch.Tensor):
                P = P.detach().cpu().numpy()
        except Exception:
            pass

        # NVIDIA Warp
        try:
            import warp as wp  # type: ignore
            if isinstance(P, wp.array):
                try:
                    # works when on CPU
                    P = P.numpy()
                except Exception:
                    # GPU path via torch
                    P = P.to_torch().cpu().numpy()
        except Exception:
            pass

        P = np.asarray(P, dtype=np.float32)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"Expected (N,3) array; got {P.shape}")
        return P
    
    def _norm_up(self, up: str) -> str:
        up = (up or "").strip().upper()
        if up not in ("Y", "Z"):
            raise ValueError(f"sim_up must be 'Y' or 'Z', got {up!r}")
        return up

    def _remap_to_stage(self, P: np.ndarray, sim_up: str) -> np.ndarray:
        """Return points expressed in the stage's up-axis frame."""
        sim_up = self._norm_up(sim_up)
        stage_up_tok = UsdGeom.GetStageUpAxis(self._stage)
        stage_up = "Y" if stage_up_tok == UsdGeom.Tokens.y else "Z"

        if sim_up == stage_up:
            return P  # no remap

        # Y-up (sim) -> Z-up (stage): +90° about X: (x, y, z) -> (x, -z, y)
        if sim_up == "Y" and stage_up == "Z":
            return np.column_stack((P[:, 0], -P[:, 2], P[:, 1]))

        # Z-up (sim) -> Y-up (stage): -90° about X: (x, y, z) -> (x, z, -y)
        # (inverse of the mapping above)
        if sim_up == "Z" and stage_up == "Y":
            return np.column_stack((P[:, 0], -P[:, 2], -P[:, 1]))

        # Should never get here
        return P

    def write_frame(self, points_now, sim_up: str = "Y"):
        if self._stage is None:
            raise RuntimeError("Call open(...) before write_frame(...)")

        # 1) Get sim-space points as float32 (N,3)
        P_sim = self._to_numpy(points_now)
        if self._num_points is not None and P_sim.shape[0] != self._num_points:
            raise ValueError(f"Got {P_sim.shape[0]} points, expected {self._num_points}")

        # 2) Remap to the stage's up-axis
        P_stage = self._remap_to_stage(P_sim, sim_up)

        
        t = float(self._frame_index)

        # 3) Author points (cast to Python floats for Boost.Python)
        pts = [_vec3f(p) for p in P_stage]
       
        self._points_attr.Set(pts, time=t)

        # 4) Extent from the same (remapped) coords
        mn = P_stage.min(axis=0)
        mx = P_stage.max(axis=0)
        self._extent_attr.Set([_vec3f(mn), _vec3f(mx)], time=t)

        # 5) Optional velocities — compute in sim space, then remap
        if self._vel_attr is not None:
            if self._prev_points is None:
                V_sim = np.zeros_like(P_sim, dtype=np.float32)
            else:
                V_sim = (P_sim - self._prev_points) * self.fps
            V_stage = self._remap_to_stage(V_sim, sim_up)
            self._vel_attr.Set([_vec3f(v) for v in V_stage], time=t)

        # 6) Advance and persist if desired
        self._stage.SetEndTimeCode(t)
        self._prev_points = P_sim
        self._frame_index += 1
        if self.flush_every_frame:
            self._stage.GetRootLayer().Save()


    def close(self):
        if self._stage:
            self._stage.GetRootLayer().Save()
        # Drop refs
        self._stage = None
        self._mesh = None
        self._points_attr = None
        self._extent_attr = None
        self._vel_attr = None
        self._prev_points = None
        self._num_points = None

