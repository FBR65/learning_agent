"""
Szenario-Manager für das Laden und Verwalten von Lernszenarien
"""

import json
from typing import List, Dict, Optional
from pathlib import Path

from .models import LearningScenario, Task, DifficultyLevel, TaskType


class ScenarioManager:
    """Verwaltet Lernszenarien und deren Laden/Speichern"""

    def __init__(self, scenarios_dir: str = "scenarios/"):
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(exist_ok=True)
        self._loaded_scenarios: Dict[str, LearningScenario] = {}

    def create_scenario_from_dict(self, scenario_data: Dict) -> LearningScenario:
        """Erstellt ein LearningScenario aus einem Dictionary"""
        # Konvertiere Tasks
        tasks = []
        for task_data in scenario_data.get("tasks", []):
            task = Task(
                id=task_data["id"],
                question=task_data["question"],
                answer=task_data["answer"],
                task_type=TaskType(task_data.get("task_type", "free_text")),
                difficulty=DifficultyLevel(task_data.get("difficulty", 1)),
                hints=task_data.get("hints", []),
                category=task_data.get("category", ""),
                prerequisites=task_data.get("prerequisites", []),
                time_limit=task_data.get("time_limit"),
                metadata=task_data.get("metadata", {}),
            )
            tasks.append(task)

        scenario = LearningScenario(
            name=scenario_data["name"],
            description=scenario_data["description"],
            version=scenario_data.get("version", "1.0"),
            tasks=tasks,
            categories=scenario_data.get("categories", []),
            progression_rules=scenario_data.get("progression_rules", {}),
            metadata=scenario_data.get("metadata", {}),
        )

        return scenario

    def load_scenario(self, scenario_name: str) -> Optional[LearningScenario]:
        """Lädt ein Szenario aus einer JSON-Datei"""
        if scenario_name in self._loaded_scenarios:
            return self._loaded_scenarios[scenario_name]

        scenario_file = self.scenarios_dir / f"{scenario_name}.json"

        if not scenario_file.exists():
            print(f"Szenario-Datei nicht gefunden: {scenario_file}")
            return None

        try:
            with open(scenario_file, "r", encoding="utf-8") as f:
                scenario_data = json.load(f)

            scenario = self.create_scenario_from_dict(scenario_data)
            self._loaded_scenarios[scenario_name] = scenario

            print(
                f"Szenario '{scenario_name}' erfolgreich geladen ({len(scenario.tasks)} Tasks)"
            )
            return scenario

        except Exception as e:
            print(f"Fehler beim Laden des Szenarios '{scenario_name}': {e}")
            return None

    def save_scenario(
        self, scenario: LearningScenario, overwrite: bool = False
    ) -> bool:
        """Speichert ein Szenario als JSON-Datei"""
        scenario_file = self.scenarios_dir / f"{scenario.name}.json"

        if scenario_file.exists() and not overwrite:
            print(f"Szenario-Datei existiert bereits: {scenario_file}")
            return False

        try:
            # Konvertiere zu Dictionary
            scenario_dict = {
                "name": scenario.name,
                "description": scenario.description,
                "version": scenario.version,
                "categories": scenario.categories,
                "progression_rules": scenario.progression_rules,
                "metadata": scenario.metadata,
                "tasks": [],
            }

            for task in scenario.tasks:
                task_dict = {
                    "id": task.id,
                    "question": task.question,
                    "answer": task.answer,
                    "task_type": task.task_type.value,
                    "difficulty": task.difficulty.value,
                    "hints": task.hints,
                    "category": task.category,
                    "prerequisites": task.prerequisites,
                    "time_limit": task.time_limit,
                    "metadata": task.metadata,
                }
                scenario_dict["tasks"].append(task_dict)

            with open(scenario_file, "w", encoding="utf-8") as f:
                json.dump(scenario_dict, f, indent=2, ensure_ascii=False)

            self._loaded_scenarios[scenario.name] = scenario
            print(f"Szenario '{scenario.name}' erfolgreich gespeichert")
            return True

        except Exception as e:
            print(f"Fehler beim Speichern des Szenarios '{scenario.name}': {e}")
            return False

    def list_scenarios(self) -> List[str]:
        """Gibt eine Liste aller verfügbaren Szenarien zurück"""
        scenario_files = list(self.scenarios_dir.glob("*.json"))
        return [f.stem for f in scenario_files]

    def get_scenario_info(self, scenario_name: str) -> Optional[Dict]:
        """Gibt grundlegende Informationen über ein Szenario zurück"""
        scenario = self.load_scenario(scenario_name)
        if not scenario:
            return None

        # Statistiken berechnen
        difficulty_counts = {}
        category_counts = {}

        for task in scenario.tasks:
            diff = task.difficulty.name
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            cat = task.category or "Uncategorized"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "name": scenario.name,
            "description": scenario.description,
            "version": scenario.version,
            "total_tasks": len(scenario.tasks),
            "categories": list(category_counts.keys()),
            "difficulty_distribution": difficulty_counts,
            "category_distribution": category_counts,
        }

    def create_sample_scenarios(self):
        """Erstellt Beispiel-Szenarien für verschiedene Lernbereiche"""

        # 1. Latein Vokabeln für Anfänger
        latin_tasks = [
            {
                "id": "latin_001",
                "question": "Was bedeutet 'puer'?",
                "answer": "Junge",
                "task_type": "translation",
                "difficulty": 1,
                "category": "Vokabeln",
                "hints": ["männlich", "Nominativ", "o-Deklination"],
            },
            {
                "id": "latin_002",
                "question": "Was bedeutet 'puella'?",
                "answer": "Mädchen",
                "task_type": "translation",
                "difficulty": 1,
                "category": "Vokabeln",
                "hints": ["weiblich", "Nominativ", "a-Deklination"],
            },
            {
                "id": "latin_003",
                "question": "Konjugiere 'amare' in der 1. Person Singular Präsens",
                "answer": "amo",
                "task_type": "code_completion",
                "difficulty": 2,
                "category": "Verben",
                "hints": ["a-Konjugation", "ich liebe"],
            },
            {
                "id": "latin_004",
                "question": "Dekliniere 'rosa' im Genitiv Singular",
                "answer": "rosae",
                "task_type": "code_completion",
                "difficulty": 2,
                "category": "Deklination",
                "hints": ["a-Deklination", "weiblich"],
            },
            {
                "id": "latin_005",
                "question": "Übersetze: 'Puella in horto ambulat'",
                "answer": "Das Mädchen wandelt im Garten",
                "task_type": "translation",
                "difficulty": 3,
                "category": "Übersetzung",
                "hints": ["3. Person Singular", "Ablativ des Ortes"],
            },
        ]

        latin_scenario = {
            "name": "latein_anfaenger",
            "description": "Latein Grundwortschatz und Grammatik für Anfänger",
            "version": "1.0",
            "categories": ["Vokabeln", "Verben", "Deklination", "Übersetzung"],
            "tasks": latin_tasks,
            "progression_rules": {
                "min_success_rate": 0.7,
                "difficulty_increase_threshold": 3,
            },
        }

        # 2. Python Programmierung für Anfänger
        python_tasks = [
            {
                "id": "python_001",
                "question": "Wie definiert man eine Variable namens 'name' mit dem Wert 'Python'?",
                "answer": "name = 'Python'",
                "task_type": "code_completion",
                "difficulty": 1,
                "category": "Variablen",
                "hints": ["Verwende das = Zeichen", "Strings in Anführungszeichen"],
            },
            {
                "id": "python_002",
                "question": "Wie gibt man 'Hallo Welt' auf der Konsole aus?",
                "answer": "print('Hallo Welt')",
                "task_type": "code_completion",
                "difficulty": 1,
                "category": "Ausgabe",
                "hints": ["print() Funktion", "String in Anführungszeichen"],
            },
            {
                "id": "python_003",
                "question": "Definiere eine Funktion namens 'greet' die einen Parameter 'name' nimmt",
                "answer": "def greet(name):",
                "task_type": "code_completion",
                "difficulty": 2,
                "category": "Funktionen",
                "hints": [
                    "def keyword",
                    "Doppelpunkt am Ende",
                    "Parameter in Klammern",
                ],
            },
            {
                "id": "python_004",
                "question": "Erstelle eine Liste mit den Zahlen 1, 2, 3",
                "answer": "[1, 2, 3]",
                "task_type": "code_completion",
                "difficulty": 2,
                "category": "Listen",
                "hints": ["eckige Klammern", "Kommas zwischen Elementen"],
            },
            {
                "id": "python_005",
                "question": "Schreibe eine for-Schleife die über die Liste [1, 2, 3] iteriert",
                "answer": "for i in [1, 2, 3]:",
                "task_type": "code_completion",
                "difficulty": 3,
                "category": "Schleifen",
                "hints": ["for keyword", "in keyword", "Doppelpunkt"],
            },
        ]

        python_scenario = {
            "name": "python_anfaenger",
            "description": "Python Programmierung Grundlagen für Anfänger",
            "version": "1.0",
            "categories": ["Variablen", "Ausgabe", "Funktionen", "Listen", "Schleifen"],
            "tasks": python_tasks,
            "progression_rules": {
                "min_success_rate": 0.8,
                "difficulty_increase_threshold": 2,
            },
        }

        # 3. Mathematik Grundlagen
        math_tasks = [
            {
                "id": "math_001",
                "question": "Was ist 5 + 3?",
                "answer": "8",
                "task_type": "free_text",
                "difficulty": 1,
                "category": "Addition",
                "hints": ["Zähle einfach zusammen"],
            },
            {
                "id": "math_002",
                "question": "Was ist 12 - 7?",
                "answer": "5",
                "task_type": "free_text",
                "difficulty": 1,
                "category": "Subtraktion",
                "hints": ["Ziehe die kleinere von der größeren Zahl ab"],
            },
            {
                "id": "math_003",
                "question": "Was ist 6 × 4?",
                "answer": "24",
                "task_type": "free_text",
                "difficulty": 2,
                "category": "Multiplikation",
                "hints": ["6 mal 4", "Addiere 6 vier Mal"],
            },
            {
                "id": "math_004",
                "question": "Was ist 15 ÷ 3?",
                "answer": "5",
                "task_type": "free_text",
                "difficulty": 2,
                "category": "Division",
                "hints": ["Wie oft passt 3 in 15?"],
            },
            {
                "id": "math_005",
                "question": "Löse: 2x + 5 = 11. Was ist x?",
                "answer": "3",
                "task_type": "free_text",
                "difficulty": 3,
                "category": "Algebra",
                "hints": ["Ziehe 5 von beiden Seiten ab", "Teile durch 2"],
            },
        ]

        math_scenario = {
            "name": "mathe_grundlagen",
            "description": "Mathematik Grundlagen: Grundrechenarten und einfache Algebra",
            "version": "1.0",
            "categories": [
                "Addition",
                "Subtraktion",
                "Multiplikation",
                "Division",
                "Algebra",
            ],
            "tasks": math_tasks,
            "progression_rules": {
                "min_success_rate": 0.75,
                "difficulty_increase_threshold": 3,
            },
        }

        # Speichere alle Szenarien
        scenarios = [latin_scenario, python_scenario, math_scenario]

        for scenario_data in scenarios:
            scenario = self.create_scenario_from_dict(scenario_data)
            self.save_scenario(scenario, overwrite=True)

        print(f"3 Beispiel-Szenarien erstellt: {[s['name'] for s in scenarios]}")
