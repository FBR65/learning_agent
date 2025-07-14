"""
Streamlit Web-Interface für den Lernagenten
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

from .models import LearnerState, AgentAction
from .scenario_manager import ScenarioManager
from .agent import LearningAgent


def init_session_state():
    """Initialisiert den Session State"""
    if "scenario_manager" not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()

    if "current_scenario" not in st.session_state:
        st.session_state.current_scenario = None

    if "learning_agent" not in st.session_state:
        st.session_state.learning_agent = None

    if "learner_state" not in st.session_state:
        st.session_state.learner_state = None

    if "current_task" not in st.session_state:
        st.session_state.current_task = None

    if "session_history" not in st.session_state:
        st.session_state.session_history = []

    if "is_learning_session_active" not in st.session_state:
        st.session_state.is_learning_session_active = False


def display_scenario_selection():
    """Zeigt die Szenario-Auswahl"""
    st.header("📚 Szenario Auswahl")

    # Lade verfügbare Szenarien
    scenarios = st.session_state.scenario_manager.list_scenarios()

    if not scenarios:
        st.warning("Keine Szenarien gefunden. Erstelle zuerst Beispiel-Szenarien.")
        if st.button("Beispiel-Szenarien erstellen"):
            st.session_state.scenario_manager.create_sample_scenarios()
            st.rerun()
        return

    # Szenario-Auswahl
    selected_scenario = st.selectbox(
        "Wähle ein Lernszenario:",
        options=scenarios,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if selected_scenario:
        # Zeige Szenario-Informationen
        scenario_info = st.session_state.scenario_manager.get_scenario_info(
            selected_scenario
        )

        if scenario_info:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(scenario_info["name"].replace("_", " ").title())
                st.write(scenario_info["description"])
                st.write(f"**Anzahl Aufgaben:** {scenario_info['total_tasks']}")
                st.write(f"**Version:** {scenario_info['version']}")

            with col2:
                st.write("**Kategorien:**")
                for category in scenario_info["categories"]:
                    st.write(f"- {category}")

                st.write("**Schwierigkeitsverteilung:**")
                for diff, count in scenario_info["difficulty_distribution"].items():
                    st.write(f"- {diff}: {count} Aufgaben")

        # Szenario laden Button
        if st.button("Szenario laden und Training starten"):
            scenario = st.session_state.scenario_manager.load_scenario(
                selected_scenario
            )
            if scenario:
                st.session_state.current_scenario = scenario
                st.session_state.learning_agent = LearningAgent(
                    scenario, algorithm="DQN"
                )
                st.success(f"Szenario '{selected_scenario}' erfolgreich geladen!")
                st.rerun()


def display_training_interface():
    """Zeigt das Training-Interface"""
    st.header("🤖 Agent Training")

    scenario = st.session_state.current_scenario
    agent = st.session_state.learning_agent

    st.write(f"**Aktuelles Szenario:** {scenario.name}")
    st.write(f"**Algorithmus:** {agent.algorithm}")

    # Training Parameter
    col1, col2 = st.columns(2)

    with col1:
        total_timesteps = st.number_input(
            "Training Timesteps",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
        )

        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            format="%.4f",
        )

    with col2:
        exploration_fraction = st.slider(
            "Exploration Fraction", min_value=0.1, max_value=0.5, value=0.3
        )

        batch_size = st.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)

    # Training starten
    if st.button("Training starten"):
        if not agent.is_trained:
            with st.spinner("Agent wird trainiert..."):
                # Erstelle Modell mit benutzerdefinierten Parametern
                agent.create_model(
                    learning_rate=learning_rate,
                    exploration_fraction=exploration_fraction,
                    batch_size=batch_size,
                )

                # Training
                episode_rewards, episode_lengths = agent.train(
                    total_timesteps=total_timesteps, save_path="models/"
                )

                # Zeige Trainings-Ergebnisse
                st.success("Training abgeschlossen!")

                # Plot Training Verlauf
                if episode_rewards:
                    fig = px.line(
                        y=episode_rewards,
                        title="Training Verlauf - Episode Rewards",
                        labels={"index": "Episode", "y": "Reward"},
                    )
                    st.plotly_chart(fig)

                st.rerun()
        else:
            st.warning("Agent ist bereits trainiert!")

    # Modell laden
    st.subheader("Vortrainiertes Modell laden")
    uploaded_file = st.file_uploader("Wähle ein Modell (.zip)", type="zip")

    if uploaded_file:
        # Speichere temporär
        with open("temp_model.zip", "wb") as f:
            f.write(uploaded_file.read())

        try:
            agent.load_model("temp_model")
            st.success("Modell erfolgreich geladen!")
            st.rerun()
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")


def display_learning_session():
    """Zeigt die Lern-Session Oberfläche"""
    st.header("🎓 Lern-Session")

    agent = st.session_state.learning_agent

    if not agent.is_trained:
        st.warning("Agent muss zuerst trainiert werden!")
        return

    # Lerner-ID eingeben
    learner_id = st.text_input("Lerner-ID", value="test_learner")

    # Session starten/stoppen
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Lern-Session starten"):
            st.session_state.learner_state = LearnerState(
                learner_id=learner_id,
                scenario_name=st.session_state.current_scenario.name,
            )
            st.session_state.is_learning_session_active = True
            st.session_state.session_history = []
            st.rerun()

    with col2:
        if st.button("Session beenden"):
            st.session_state.is_learning_session_active = False
            st.rerun()

    if not st.session_state.is_learning_session_active:
        return

    learner_state = st.session_state.learner_state

    # Zeige aktuellen Lerner-Zustand
    display_learner_dashboard(learner_state)

    # Hole nächste Aktion vom Agenten
    try:
        agent_action = agent.get_action(learner_state)
        st.session_state.current_action = agent_action

        # Zeige Agent-Aktion
        display_agent_action(agent_action)

    except Exception as e:
        st.error(f"Fehler beim Abrufen der Agent-Aktion: {e}")


def display_learner_dashboard(learner_state: LearnerState):
    """Zeigt das Lerner-Dashboard"""
    st.subheader("📊 Dein Fortschritt")

    # Erfolgsraten pro Kategorie
    categories = st.session_state.current_scenario.categories
    if categories:
        success_rates = []
        for category in categories:
            rate = learner_state.get_success_rate(category)
            success_rates.append({"Kategorie": category, "Erfolgsrate": rate})

        if success_rates:
            df = pd.DataFrame(success_rates)
            fig = px.bar(
                df, x="Kategorie", y="Erfolgsrate", title="Erfolgsrate nach Kategorien"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Statistiken
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_correct = sum(learner_state.correct_answers.values())
        st.metric("Richtige Antworten", total_correct)

    with col2:
        total_incorrect = sum(learner_state.incorrect_answers.values())
        st.metric("Falsche Antworten", total_incorrect)

    with col3:
        st.metric("Korrekt hintereinander", learner_state.consecutive_correct)

    with col4:
        total_hints = sum(learner_state.hints_used.values())
        st.metric("Hinweise verwendet", total_hints)


def display_agent_action(agent_action: AgentAction):
    """Zeigt die Agent-Aktion und verarbeitet Lerner-Input"""
    st.subheader("🤖 Nächste Aufgabe")

    action_type = agent_action.action_type
    params = agent_action.parameters

    if action_type.startswith("present_"):
        # Zeige Aufgabe
        if "question" in params:
            st.write(f"**Frage:** {params['question']}")

            # Schwierigkeitsanzeige
            difficulty_map = {1: "Leicht", 2: "Mittel", 3: "Schwer", 4: "Experte"}
            if "difficulty" in params:
                difficulty_name = difficulty_map.get(params["difficulty"], "Unbekannt")
                st.badge(f"Schwierigkeit: {difficulty_name}")

        # Antwort-Eingabe
        user_answer = st.text_input("Deine Antwort:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Antwort abschicken"):
                process_user_answer(user_answer, params.get("task_id"))

        with col2:
            if st.button("Hinweis anfordern"):
                show_hint(params.get("task_id"))

    elif action_type == "give_hint":
        st.info("💡 Der Agent gibt dir einen Hinweis...")
        if st.session_state.current_task:
            task = st.session_state.current_task
            if task.hints:
                hint = task.hints[0]  # Zeige ersten Hinweis
                st.write(f"**Hinweis:** {hint}")

    elif action_type == "provide_feedback":
        st.success("✅ Gut gemacht! Weiter zur nächsten Aufgabe.")
        if st.button("Weiter"):
            st.rerun()

    elif action_type == "reduce_difficulty":
        st.warning("🔽 Der Agent reduziert die Schwierigkeit.")
        if st.button("Weiter"):
            st.rerun()


def process_user_answer(user_answer: str, task_id: str):
    """Verarbeitet die Benutzer-Antwort"""
    if not user_answer.strip():
        st.warning("Bitte gib eine Antwort ein.")
        return

    # Finde die aktuelle Aufgabe
    task = st.session_state.current_scenario.get_task_by_id(task_id)
    if not task:
        st.error("Aufgabe nicht gefunden.")
        return

    # Prüfe Antwort
    is_correct = user_answer.strip().lower() == task.answer.strip().lower()

    # Update Lerner-Zustand
    learner_state = st.session_state.learner_state
    category = task.category or "general"

    if is_correct:
        learner_state.correct_answers[category] = (
            learner_state.correct_answers.get(category, 0) + 1
        )
        learner_state.consecutive_correct += 1
        learner_state.consecutive_incorrect = 0

        st.success(f"✅ Richtig! Die Antwort ist: {task.answer}")
    else:
        learner_state.incorrect_answers[category] = (
            learner_state.incorrect_answers.get(category, 0) + 1
        )
        learner_state.consecutive_incorrect += 1
        learner_state.consecutive_correct = 0

        st.error(f"❌ Falsch. Die richtige Antwort ist: {task.answer}")

    # Füge zur Session-Historie hinzu
    st.session_state.session_history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "question": task.question,
            "user_answer": user_answer,
            "correct_answer": task.answer,
            "is_correct": is_correct,
        }
    )

    # Kurz warten und neu laden
    time.sleep(1)
    st.rerun()


def show_hint(task_id: str):
    """Zeigt einen Hinweis für die aktuelle Aufgabe"""
    task = st.session_state.current_scenario.get_task_by_id(task_id)
    if task and task.hints:
        # Update Hinweis-Statistik
        learner_state = st.session_state.learner_state
        category = task.category or "general"
        learner_state.hints_used[category] = (
            learner_state.hints_used.get(category, 0) + 1
        )

        # Zeige Hinweis
        hint = task.hints[0]  # Zeige ersten Hinweis
        st.info(f"💡 Hinweis: {hint}")
    else:
        st.warning("Keine Hinweise für diese Aufgabe verfügbar.")


def display_analytics():
    """Zeigt Analytics und Statistiken"""
    st.header("📈 Analytics")

    if not st.session_state.session_history:
        st.info("Noch keine Lern-Session Daten verfügbar.")
        return

    # Konvertiere zu DataFrame
    df = pd.DataFrame(st.session_state.session_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Grundlegende Statistiken
    col1, col2, col3 = st.columns(3)

    with col1:
        total_questions = len(df)
        st.metric("Gesamt Fragen", total_questions)

    with col2:
        correct_answers = df["is_correct"].sum()
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        st.metric("Genauigkeit", f"{accuracy:.1%}")

    with col3:
        st.metric("Richtige Antworten", correct_answers)

    # Zeitlicher Verlauf
    if len(df) > 1:
        fig = px.line(df, x="timestamp", y="is_correct", title="Antworten über Zeit")
        st.plotly_chart(fig, use_container_width=True)

    # Session Details
    st.subheader("Session Details")
    st.dataframe(
        df[["timestamp", "question", "user_answer", "correct_answer", "is_correct"]]
    )


def main():
    """Hauptfunktion der Streamlit App"""
    st.set_page_config(
        page_title="Universeller Lernagent", page_icon="🤖", layout="wide"
    )

    st.title("🤖 Universeller Lernagent")
    st.write("Personalisierter Lernassistent mit adaptiver Schwierigkeitsanpassung")

    # Initialisiere Session State
    init_session_state()

    # Sidebar Navigation
    st.sidebar.title("Navigation")

    if st.session_state.current_scenario is None:
        st.sidebar.info("Wähle zuerst ein Lernszenario")
        page = "Szenario Auswahl"
    else:
        page = st.sidebar.radio(
            "Seite wählen:",
            ["Szenario Auswahl", "Agent Training", "Lern-Session", "Analytics"],
        )

    # Zeige aktuelle Szenario-Info in Sidebar
    if st.session_state.current_scenario:
        st.sidebar.success(f"Szenario: {st.session_state.current_scenario.name}")

        if (
            st.session_state.learning_agent
            and st.session_state.learning_agent.is_trained
        ):
            st.sidebar.success("Agent ist trainiert ✓")
        else:
            st.sidebar.warning("Agent muss trainiert werden")

    # Routing
    if page == "Szenario Auswahl":
        display_scenario_selection()
    elif page == "Agent Training":
        if st.session_state.current_scenario:
            display_training_interface()
        else:
            st.warning("Bitte wähle zuerst ein Szenario.")
    elif page == "Lern-Session":
        if st.session_state.current_scenario:
            display_learning_session()
        else:
            st.warning("Bitte wähle zuerst ein Szenario.")
    elif page == "Analytics":
        display_analytics()


if __name__ == "__main__":
    main()
