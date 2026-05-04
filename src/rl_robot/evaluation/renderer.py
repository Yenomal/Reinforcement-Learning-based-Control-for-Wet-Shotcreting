from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


class EvalRenderer:
    """Matplotlib-based real-time renderer for evaluation rollout."""

    def __init__(
        self,
        surface_scene: Dict[str, np.ndarray],
        render_pause: float = 0.05,
        episode_pause: float = 0.8,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from matplotlib.widgets import Button
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for interactive evaluation rendering."
            ) from exc

        self.plt = plt
        self.Button = Button
        self.render_pause = render_pause
        self.episode_pause = episode_pause
        self.started = False

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(bottom=0.14)

        x_grid = surface_scene["x_grid"]
        y_grid = surface_scene["y_grid"]
        z_grid = surface_scene["z_grid"]
        color_grid = surface_scene["color_grid"]

        norm = Normalize(vmin=float(color_grid.min()), vmax=float(color_grid.max()))
        facecolors = cm.get_cmap("terrain")(norm(color_grid))
        self.ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.75,
        )

        self.point_line, = self.ax.plot(
            [], [], [], color="crimson", linewidth=2.5, label="point_path"
        )
        self.current_point_artist, = self.ax.plot(
            [], [], [], marker="o", color="crimson", markersize=6, linestyle=""
        )

        self.start_artist = None
        self.goal_artist = None
        self.start_link = None
        self.goal_link = None
        self.status_text = self.ax.text2D(
            0.02,
            0.98,
            "Waiting to start...",
            transform=self.ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.button_ax = self.fig.add_axes([0.42, 0.03, 0.16, 0.06])
        self.start_button = self.Button(self.button_ax, "Start")
        self.start_button.on_clicked(self._handle_start)

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("MathEnv Evaluation")
        self.ax.legend(loc="upper right")

        x_span = float(x_grid.max() - x_grid.min())
        y_span = float(y_grid.max() - y_grid.min())
        z_span = float(z_grid.max() - z_grid.min())
        self.ax.set_box_aspect(
            (max(x_span, 1e-3), max(y_span, 1e-3), max(z_span, 1e-3))
        )

        self.fig.tight_layout()
        self.plt.ion()
        self.plt.show(block=False)

    def _handle_start(self, _event: object) -> None:
        """Start the evaluation rollout after the button is pressed."""
        self.started = True
        self.start_button.label.set_text("Running...")
        self.status_text.set_text("Evaluation started.\nPreparing first rollout...")
        self.fig.canvas.draw_idle()

    def wait_for_start(self) -> None:
        """Block in the UI loop until the Start button is pressed."""
        while not self.started:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def wait_for_start_with_idle(
        self,
        on_idle: Optional[Callable[[], None]] = None,
    ) -> None:
        """Block until Start is pressed while optionally ticking other viewers."""
        while not self.started:
            if on_idle is not None:
                on_idle()
            self.fig.canvas.draw_idle()
            self.plt.pause(0.05)

    def reset_episode(
        self,
        episode_index: int,
        total_episodes: int,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ) -> None:
        """Clear the old trajectory and mark the new task."""
        self.point_line.set_data_3d([], [], [])
        self.current_point_artist.set_data_3d([], [], [])

        if self.start_artist is not None:
            self.start_artist.remove()
        if self.goal_artist is not None:
            self.goal_artist.remove()
        if self.start_link is not None:
            self.start_link.remove()
            self.start_link = None
        if self.goal_link is not None:
            self.goal_link.remove()
            self.goal_link = None

        self.start_artist = self.ax.scatter(
            [start_point[0]],
            [start_point[1]],
            [start_point[2]],
            c="limegreen",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            label="start",
            depthshade=False,
        )
        self.goal_artist = self.ax.scatter(
            [goal_point[0]],
            [goal_point[1]],
            [goal_point[2]],
            c="gold",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            label="goal",
            depthshade=False,
        )

        self.status_text.set_text(
            f"Episode {episode_index}/{total_episodes}\nWaiting for rollout..."
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.render_pause)

    def update(
        self,
        episode_index: int,
        total_episodes: int,
        point_path: np.ndarray,
        reward: float,
        episode_return: float,
        step: int,
        goal_distance: float,
    ) -> None:
        """Refresh the trajectory lines and overlay metrics."""
        self.point_line.set_data_3d(
            point_path[:, 0], point_path[:, 1], point_path[:, 2]
        )
        self.current_point_artist.set_data_3d(
            [point_path[-1, 0]],
            [point_path[-1, 1]],
            [point_path[-1, 2]],
        )

        self.status_text.set_text(
            "\n".join(
                [
                    f"Episode {episode_index}/{total_episodes}",
                    f"Step: {step}",
                    f"Reward: {reward:.3f}",
                    f"Return: {episode_return:.3f}",
                    f"Goal dist: {goal_distance:.3f}",
                ]
            )
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.render_pause)

    def pause_between_episodes(
        self,
        episode_index: int,
        total_episodes: int,
        episode_return: float,
        success: bool,
        min_goal_distance: float,
    ) -> None:
        """Show one short episode-end status before the next task starts."""
        self.status_text.set_text(
            "\n".join(
                [
                    f"Episode {episode_index}/{total_episodes} finished",
                    f"Return: {episode_return:.3f}",
                    f"Success: {success}",
                    f"Min dist: {min_goal_distance:.4f}",
                ]
            )
        )
        self.fig.canvas.draw_idle()
        self.plt.pause(self.episode_pause)

    def finalize(self) -> None:
        """Keep the last frame visible after evaluation."""
        self.start_button.label.set_text("Done")
        self.fig.canvas.draw_idle()
        self.plt.ioff()
        self.plt.show()
