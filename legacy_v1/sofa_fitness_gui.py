import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sofa_shape_gui import SofaWindow
from sofa_fitness import evaluate_sofa


class FitnessWindow(SofaWindow):
    def __init__(self):
        self.fitness_result = None
        self.fitness_error = ""
        super().__init__()

        self.fig.set_size_inches(13, 8.5)
        self.fig.canvas.manager.set_window_title("Moving Sofa - Fitness Controls")
        self.ax.set_title("Moving Sofa - Shape and Fitness")
        self.ax.set_position([0.06, 0.43, 0.60, 0.49])
        self.status.set_position((0.08, 0.35))

        self.fig.text(0.73, 0.245, "Wall-overlap penalty weight", fontsize=10)
        self.weight_slider = Slider(
            self.fig.add_axes([0.73, 0.205, 0.20, 0.025]),
            "", 1.0, 50.0, valinit=10.0, valstep=0.5, valfmt="%.1f",
        )
        self.fig.text(
            0.73, 0.17,
            "Choose settings, then apply.\n"
            "Play previews the evaluated route.",
            fontsize=9, color="#64748b",
        )
        self.apply_button.ax.set_position([0.73, 0.09, 0.22, 0.05])
        self.apply_button.label.set_text("Apply + evaluate")

        self.fig.text(0.08, 0.245, "FITNESS", fontsize=11, weight="bold")
        self.score_text = self.fig.text(
            0.08, 0.225, "", fontsize=24, weight="bold", va="top",
        )
        self.verdict_text = self.fig.text(0.08, 0.172, "", fontsize=10)
        self.fig.text(0.08, 0.145, "Higher is better.",
                      fontsize=9, color="#64748b")
        self.metrics_text = self.fig.text(
            0.35, 0.245, "", fontsize=10, va="top", linespacing=1.4,
        )

        self.weight_slider.on_changed(self.settings_changed)
        self.show_fitness()
        self.fig.sofa_window = self

    def settings_changed(self, _=None):
        self.fitness_result = None
        self.fitness_error = ""
        super().settings_changed(_)
        self.show_fitness()

    def apply_settings(self, _=None):
        self.fitness_result = None
        self.fitness_error = ""
        super().apply_settings(_)

        if self.pending:
            self.fitness_error = self.status.get_text()
            self.show_fitness()
            return

        weight = self.weight_slider.val if hasattr(self, "weight_slider") else 10.0
        try:
            result = evaluate_sofa(self.sofa, self.poses, outside_weight=weight)
            if not result["valid_shape"]:
                raise ValueError(result["reason"])
            self.fitness_result = result
            self.applied_weight = weight
        except ValueError as error:
            self.fitness_error = str(error)
            self.pending = True
            self.play_button.set_active(False)

        self.show_fitness()

    def show_fitness(self):
        if not hasattr(self, "score_text"):
            return

        if self.fitness_error:
            color = "#b91c1c"
            self.score_text.set_text("ERROR")
            self.verdict_text.set_text("Adjust settings and apply again")
            self.metrics_text.set_text(self.fitness_error)
        elif self.pending or self.fitness_result is None:
            color = "#64748b"
            self.score_text.set_text("Pending")
            self.verdict_text.set_text("Settings have changed")
            self.metrics_text.set_text(
                "Click Apply + evaluate to calculate\n"
                "fitness for the selected settings."
            )
        else:
            result = self.fitness_result
            fits = result["sampled_feasible"]
            color = "#166534" if fits else "#b91c1c"
            self.score_text.set_text(f"{result['fitness']:.5f}")
            self.verdict_text.set_text(
                "No sampled collisions" if fits else "Sampled collisions detected"
            )
            self.metrics_text.set_text(
                f"Area: {result['area']:.5f} square units\n"
                f"Collisions: {result['colliding_samples']} / "
                f"{result['samples_checked']} ({result['collision_fraction']:.1%})\n"
                f"Mean outside area: {result['mean_outside_area']:.5f}\n"
                f"Max outside area: {result['max_outside_area']:.5f}\n"
                f"Applied penalty weight: {self.applied_weight:.1f}"
            )

        self.score_text.set_color(color)
        self.verdict_text.set_color(color)
        self.fig.canvas.draw_idle()


def main():
    window = FitnessWindow()
    plt.show()
    return window


if __name__ == "__main__":
    main()
