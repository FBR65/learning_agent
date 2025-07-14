"""
Hauptskript für den Universellen Lernagenten
"""

import argparse
import sys
from pathlib import Path

# Füge src zum Python Path hinzu
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import nach PYTHONPATH Setup
from learning_agent.scenario_manager import ScenarioManager
from learning_agent.agent import LearningAgent
from learning_agent.api import launch_api


def create_sample_scenarios():
    """Erstellt Beispiel-Szenarien"""
    manager = ScenarioManager()
    manager.create_sample_scenarios()
    print("Beispiel-Szenarien erstellt!")


def train_agent(scenario_name: str, algorithm: str = "DQN", timesteps: int = 10000):
    """Trainiert einen Agenten für ein bestimmtes Szenario"""
    manager = ScenarioManager()
    scenario = manager.load_scenario(scenario_name)

    if not scenario:
        print(f"Szenario '{scenario_name}' nicht gefunden!")
        return

    print(f"Trainiere {algorithm} Agent für Szenario: {scenario.name}")
    print(f"Anzahl Tasks: {len(scenario.tasks)}")
    print(f"Kategorien: {', '.join(scenario.categories)}")

    agent = LearningAgent(scenario, algorithm=algorithm)
    episode_rewards, episode_lengths = agent.train(
        total_timesteps=timesteps, save_path="models/"
    )

    print("Training abgeschlossen!")

    # Evaluation
    results = agent.evaluate(n_episodes=5)
    print("Evaluation Ergebnisse:")
    print(f"  Durchschnittliche Belohnung: {results['mean_reward']:.2f}")
    print(f"  Durchschnittliche Erfolgsrate: {results['mean_success_rate']:.2%}")


def list_scenarios():
    """Listet alle verfügbaren Szenarien auf"""
    manager = ScenarioManager()
    scenarios = manager.list_scenarios()

    if not scenarios:
        print("Keine Szenarien gefunden.")
        print("Erstelle Beispiel-Szenarien mit: python main.py --create-scenarios")
        return

    print("Verfügbare Szenarien:")
    for scenario_name in scenarios:
        info = manager.get_scenario_info(scenario_name)
        if info:
            print(
                f"  - {scenario_name}: {info['description']} ({info['total_tasks']} Tasks)"
            )


def run_web_ui():
    """Startet das FastAPI Web-Interface"""
    from learning_agent.api import launch_api

    print("Starte Web-Interface...")
    print("Die Anwendung wird in deinem Browser geöffnet: http://localhost:8001")

    launch_api(host="0.0.0.0", port=8001)


def main():
    """Hauptfunktion mit CLI-Interface"""
    parser = argparse.ArgumentParser(
        description="Universeller Lernagent mit adaptiver Schwierigkeitsanpassung"
    )

    parser.add_argument(
        "--create-scenarios", action="store_true", help="Erstellt Beispiel-Szenarien"
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Listet alle verfügbaren Szenarien auf",
    )

    parser.add_argument(
        "--train", type=str, help="Trainiert einen Agenten für das angegebene Szenario"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["DQN", "A2C", "PPO"],
        default="DQN",
        help="RL-Algorithmus für das Training (default: DQN)",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Anzahl der Trainings-Timesteps (default: 10000)",
    )

    parser.add_argument("--web", action="store_true", help="Startet das Web-Interface")

    args = parser.parse_args()

    # Standardverhalten: Web-Interface starten wenn keine Argumente
    if len(sys.argv) == 1:
        run_web_ui()
        return

    if args.create_scenarios:
        create_sample_scenarios()

    elif args.list_scenarios:
        list_scenarios()

    elif args.train:
        train_agent(args.train, args.algorithm, args.timesteps)

    elif args.web:
        run_web_ui()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
