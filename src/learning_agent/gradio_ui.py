"""
Gradio Web-Interface für den Lernagenten
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

from .models import LearnerState
from .scenario_manager import ScenarioManager
from .agent import LearningAgent


class LearningAgentUI:
    """Gradio-basierte Benutzeroberfläche für den Lernagenten"""

    def __init__(self):
        self.scenario_manager = ScenarioManager()
        self.current_scenario = None
        self.learning_agent = None
        self.learner_state = None
        self.current_task = None
        self.session_data = []

    def load_scenario(self, scenario_name: str) -> tuple:
        """Lädt ein Szenario"""
        if not scenario_name:
            return "❌ Bitte wähle ein Szenario", ""

        scenario = self.scenario_manager.load_scenario(scenario_name)
        if scenario:
            self.current_scenario = scenario
            self.learning_agent = LearningAgent(scenario)
            self.learner_state = LearnerState()

            info = (
                f"**Szenario:** {scenario.name}\n"
                f"**Beschreibung:** {scenario.description}\n"
                f"**Kategorien:** {', '.join(scenario.categories)}\n"
                f"**Anzahl Tasks:** {len(scenario.tasks)}\n"
                f"**Schwierigkeitsgrade:** {min(t.difficulty for t in scenario.tasks)} - {max(t.difficulty for t in scenario.tasks)}"
            )

            return ("✅ Szenario erfolgreich geladen", info)

        return "❌ Szenario konnte nicht geladen werden", ""

    def train_agent(
        self, algorithm: str, timesteps: int, progress=gr.Progress()
    ) -> str:
        """Trainiert den Agenten"""
        if not self.current_scenario:
            return "❌ Bitte lade zuerst ein Szenario"

        try:
            progress(0, desc="Initialisiere Training...")

            self.learning_agent = LearningAgent(
                self.current_scenario, algorithm=algorithm
            )

            progress(0.1, desc="Starte Training...")

            # Training mit Progress Updates
            episode_rewards, episode_lengths = self.learning_agent.train(
                total_timesteps=timesteps,
                save_path="models/",
                progress_callback=lambda x: progress(
                    0.1 + 0.8 * x, desc=f"Training... {x:.1%}"
                ),
            )

            progress(0.9, desc="Evaluiere Modell...")

            # Evaluation
            results = self.learning_agent.evaluate(n_episodes=5)

            progress(1.0, desc="Training abgeschlossen!")

            return (
                f"✅ Training abgeschlossen!\n\n"
                f"**Algorithmus:** {algorithm}\n"
                f"**Timesteps:** {timesteps}\n"
                f"**Durchschnittliche Belohnung:** {results['mean_reward']:.2f}\n"
                f"**Erfolgsrate:** {results['mean_success_rate']:.2%}\n"
                f"**Episoden:** {len(episode_rewards)}"
            )

        except Exception as e:
            return f"❌ Training fehlgeschlagen: {str(e)}"

    def start_learning_session(self) -> tuple:
        """Startet eine neue Lernsession"""
        if not self.current_scenario or not self.learning_agent:
            return (
                "❌ Bitte lade ein Szenario und trainiere den Agent",
                "",
                gr.update(visible=False),
            )

        if not self.learning_agent.is_trained:
            return "❌ Agent ist noch nicht trainiert", "", gr.update(visible=False)

        # Reset learner state
        self.learner_state = LearnerState()
        self.session_data = []

        # Erste Aktion vom Agent
        action = self.learning_agent.get_action(self.learner_state)
        self.current_task = action

        if action.action_type == "present_task":
            task = next(
                (t for t in self.current_scenario.tasks if t.id == action.task_id), None
            )
            if task:
                return (
                    "🎯 Neue Lernsession gestartet!",
                    f"**Frage:** {task.question}",
                    gr.update(visible=True),
                )

        return "❌ Fehler beim Starten der Session", "", gr.update(visible=False)

    def submit_answer(self, answer: str) -> tuple:
        """Verarbeitet eine Antwort"""
        if not answer.strip():
            return "❓ Bitte gib eine Antwort ein", "", "", gr.update(visible=True)

        if not self.current_task or not self.learning_agent:
            return "❌ Keine aktive Aufgabe", "", "", gr.update(visible=True)

        # Finde die aktuelle Aufgabe
        task = next(
            (
                t
                for t in self.current_scenario.tasks
                if t.id == self.current_task.task_id
            ),
            None,
        )
        if not task:
            return "❌ Aufgabe nicht gefunden", "", "", gr.update(visible=True)

        # Bewerte Antwort
        is_correct = answer.strip().lower() == task.answer.lower()

        # Aktualisiere Learner State
        if is_correct:
            self.learner_state.correct_answers += 1
            self.learner_state.topic_performance[task.category] = (
                self.learner_state.topic_performance.get(task.category, 0) + 1
            )
            feedback = f"✅ **Richtig!** Die Antwort '{task.answer}' ist korrekt."
        else:
            self.learner_state.incorrect_answers += 1
            feedback = f"❌ **Falsch.** Die richtige Antwort wäre: '{task.answer}'"

        # Session-Daten aktualisieren
        self.session_data.append(
            {
                "timestamp": datetime.now(),
                "task_id": task.id,
                "question": task.question,
                "user_answer": answer,
                "correct_answer": task.answer,
                "is_correct": is_correct,
                "difficulty": task.difficulty,
                "category": task.category,
            }
        )

        # Nächste Aktion vom Agent
        next_action = self.learning_agent.get_action(self.learner_state)
        self.current_task = next_action

        if next_action.action_type == "present_task":
            next_task = next(
                (t for t in self.current_scenario.tasks if t.id == next_action.task_id),
                None,
            )
            if next_task:
                next_question = f"**Nächste Frage:** {next_task.question}"
            else:
                next_question = "Keine weitere Aufgabe verfügbar"
        elif next_action.action_type == "give_hint":
            next_question = f"**Hinweis:** {next_action.content}"
        else:
            next_question = "Session beendet"

        # Statistiken
        stats = self._get_session_stats()

        return feedback, next_question, stats, gr.update(visible=True)

    def _get_session_stats(self) -> str:
        """Erstellt Session-Statistiken"""
        if not self.session_data:
            return "Noch keine Antworten"

        total = len(self.session_data)
        correct = sum(1 for d in self.session_data if d["is_correct"])
        accuracy = correct / total * 100

        # Performance nach Kategorie
        categories = {}
        for data in self.session_data:
            cat = data["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0}
            categories[cat]["total"] += 1
            if data["is_correct"]:
                categories[cat]["correct"] += 1

        stats = f"""
        **Session Statistiken:**
        - Gesamt beantwortet: {total}
        - Richtig: {correct}
        - Genauigkeit: {accuracy:.1f}%
        
        **Performance nach Kategorie:**
        """

        for cat, data in categories.items():
            cat_accuracy = data["correct"] / data["total"] * 100
            stats += (
                f"\n- {cat}: {data['correct']}/{data['total']} ({cat_accuracy:.1f}%)"
            )

        return stats

    def get_available_scenarios(self) -> list:
        """Gibt verfügbare Szenarien zurück"""
        scenarios = self.scenario_manager.list_scenarios()
        if not scenarios:
            # Erstelle Beispiel-Szenarien falls keine vorhanden
            self.scenario_manager.create_sample_scenarios()
            scenarios = self.scenario_manager.list_scenarios()

        return scenarios

    def create_progress_chart(self) -> Optional[go.Figure]:
        """Erstellt ein Fortschritts-Diagramm"""
        if not self.session_data:
            return None

        df = pd.DataFrame(self.session_data)
        df["cumulative_correct"] = df["is_correct"].cumsum()
        df["question_number"] = range(1, len(df) + 1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["question_number"],
                y=df["cumulative_correct"],
                mode="lines+markers",
                name="Richtige Antworten (kumulativ)",
                line=dict(color="green"),
            )
        )

        fig.update_layout(
            title="Lernfortschritt",
            xaxis_title="Frage Nummer",
            yaxis_title="Richtige Antworten (kumulativ)",
            height=400,
        )

        return fig


def create_interface() -> gr.Blocks:
    """Erstellt die Gradio-Oberfläche"""
    ui = LearningAgentUI()

    with gr.Blocks(title="🤖 Universeller Lernagent", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 Universeller Lernagent")
        gr.Markdown(
            "Personalisierter Lernassistent mit adaptiver Schwierigkeitsanpassung"
        )

        with gr.Tabs():
            # Tab 1: Szenario Auswahl
            with gr.TabItem("📚 Szenario Auswahl"):
                gr.Markdown("## Wähle ein Lernszenario")

                scenario_dropdown = gr.Dropdown(
                    choices=ui.get_available_scenarios(),
                    label="Verfügbare Szenarien",
                    interactive=True,
                )

                load_btn = gr.Button("Szenario laden", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
                scenario_info = gr.Markdown()

                load_btn.click(
                    fn=ui.load_scenario,
                    inputs=[scenario_dropdown],
                    outputs=[load_status, scenario_info],
                )

            # Tab 2: Agent Training
            with gr.TabItem("🎯 Agent Training", visible=False):
                gr.Markdown("## Trainiere den RL-Agent")

                with gr.Row():
                    algorithm_radio = gr.Radio(
                        choices=["DQN", "A2C", "PPO"], value="DQN", label="Algorithmus"
                    )
                    timesteps_slider = gr.Slider(
                        minimum=1000,
                        maximum=50000,
                        value=10000,
                        step=1000,
                        label="Training Timesteps",
                    )

                train_btn = gr.Button("Training starten", variant="primary")
                training_status = gr.Textbox(label="Training Status", interactive=False)

                train_btn.click(
                    fn=ui.train_agent,
                    inputs=[algorithm_radio, timesteps_slider],
                    outputs=[training_status],
                )

            # Tab 3: Lernsession
            with gr.TabItem("🎓 Lernsession"):
                gr.Markdown("## Starte deine Lernsession")

                with gr.Row():
                    start_session_btn = gr.Button(
                        "Neue Session starten", variant="primary"
                    )
                    session_status = gr.Textbox(
                        label="Session Status", interactive=False
                    )

                with gr.Group(visible=False) as learning_group:
                    current_question = gr.Markdown()

                    with gr.Row():
                        answer_input = gr.Textbox(
                            label="Deine Antwort",
                            placeholder="Gib hier deine Antwort ein...",
                            interactive=True,
                        )
                        submit_btn = gr.Button("Antwort senden", variant="secondary")

                    feedback = gr.Markdown()
                    session_stats = gr.Markdown()

                start_session_btn.click(
                    fn=ui.start_learning_session,
                    outputs=[session_status, current_question, learning_group],
                )

                submit_btn.click(
                    fn=ui.submit_answer,
                    inputs=[answer_input],
                    outputs=[feedback, current_question, session_stats, answer_input],
                )

                # Enter-Taste für Submit
                answer_input.submit(
                    fn=ui.submit_answer,
                    inputs=[answer_input],
                    outputs=[feedback, current_question, session_stats, answer_input],
                )

            # Tab 4: Analytics
            with gr.TabItem("📊 Analytics"):
                gr.Markdown("## Lernfortschritt und Statistiken")

                refresh_btn = gr.Button("Statistiken aktualisieren")
                progress_chart = gr.Plot(label="Fortschritts-Diagramm")

                refresh_btn.click(fn=ui.create_progress_chart, outputs=[progress_chart])

    return app


def launch_ui(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """Startet die Gradio-Oberfläche"""
    app = create_interface()
    app.launch(server_name=host, server_port=port, share=share, show_error=True)


if __name__ == "__main__":
    launch_ui()
