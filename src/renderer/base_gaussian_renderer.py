class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None  # Expected to be a GaussianSet instance
        self._reduce_updates = True

    @property
    def reduce_updates(self) -> bool:
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val: bool):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self) -> None:
        print("VSync is not supported")

    def update_gaussian_data(self, gaussian_set, full_update: bool = False) -> None:
        raise NotImplementedError()

    def sort_and_update(self) -> None:
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float) -> None:
        raise NotImplementedError()

    def set_render_mode(self, mod: int) -> None:
        raise NotImplementedError()

    def update_camera_pose(self) -> None:
        raise NotImplementedError()

    def update_camera_intrin(self) -> None:
        raise NotImplementedError()

    def draw(self) -> None:
        raise NotImplementedError()

    def set_model_matrix(self, model_mat) -> None:
        raise NotImplementedError()

    def reset_gaussians(self) -> None:
        raise NotImplementedError()

    def set_render_resolution(self, w: int, h: int) -> None:
        raise NotImplementedError()