"""
Multi-Format Scenario Importer
Unterstützt: CSV, JSON, XML, XLS, XLSX
"""

import json
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from io import StringIO, BytesIO
import csv

from .models import LearningScenario, Task, TaskType, DifficultyLevel


class ScenarioImporter:
    """Importiert Szenarien aus verschiedenen Dateiformaten"""

    def __init__(self):
        self.supported_formats = [".json", ".csv", ".xml", ".xls", ".xlsx"]

    def import_scenario(
        self,
        file_path: Union[str, Path, BytesIO],
        file_format: str = None,
        scenario_name: str = None,
        scenario_description: str = None,
    ) -> Optional[LearningScenario]:
        """
        Importiert ein Szenario aus einer Datei

        Args:
            file_path: Pfad zur Datei oder BytesIO-Objekt
            file_format: Format (json, csv, xml, xls, xlsx) - wird automatisch erkannt falls None
            scenario_name: Name für das Szenario (falls nicht in Datei)
            scenario_description: Beschreibung (falls nicht in Datei)
        """
        try:
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
                if not file_format:
                    file_format = file_path.suffix.lower()
            elif not file_format:
                raise ValueError(
                    "file_format muss angegeben werden für BytesIO-Objekte"
                )

            # Format normalisieren
            if not file_format.startswith("."):
                file_format = f".{file_format}"

            if file_format not in self.supported_formats:
                raise ValueError(f"Nicht unterstütztes Format: {file_format}")

            # Entsprechende Import-Methode aufrufen
            if file_format == ".json":
                data = self._import_json(file_path)
            elif file_format == ".csv":
                data = self._import_csv(file_path, scenario_name, scenario_description)
            elif file_format == ".xml":
                data = self._import_xml(file_path, scenario_name, scenario_description)
            elif file_format in [".xls", ".xlsx"]:
                data = self._import_excel(
                    file_path, scenario_name, scenario_description
                )
            else:
                raise ValueError(
                    f"Handler für Format {file_format} nicht implementiert"
                )

            # Validierung und Konvertierung
            return self._create_scenario_from_data(data)

        except Exception as e:
            print(f"Fehler beim Importieren: {e}")
            return None

    def _import_json(self, file_path: Union[Path, BytesIO]) -> Dict[str, Any]:
        """Importiert JSON-Format"""
        if isinstance(file_path, BytesIO):
            data = json.load(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return data

    def _import_csv(
        self,
        file_path: Union[Path, BytesIO],
        scenario_name: str = None,
        scenario_description: str = None,
    ) -> Dict[str, Any]:
        """
        Importiert CSV-Format
        Erwartete Spalten: id, question, answer, task_type, difficulty, category, hints
        """
        if isinstance(file_path, BytesIO):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)

        # Standard-Werte setzen
        name = (
            scenario_name
            or f"imported_scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        description = scenario_description or "Importiertes Szenario aus CSV"

        # Tasks konvertieren
        tasks = []
        categories = set()

        for _, row in df.iterrows():
            # Hints verarbeiten (kann String mit Semikolon-Trennung sein)
            hints = []
            if pd.notna(row.get("hints", "")):
                hints_str = str(row["hints"])
                if ";" in hints_str:
                    hints = [h.strip() for h in hints_str.split(";")]
                elif "," in hints_str:
                    hints = [h.strip() for h in hints_str.split(",")]
                else:
                    hints = [hints_str.strip()]

            task_data = {
                "id": str(row.get("id", f"task_{len(tasks) + 1:03d}")),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "task_type": str(row.get("task_type", "free_text")),
                "difficulty": int(row.get("difficulty", 1)),
                "category": str(row.get("category", "General")),
                "hints": hints,
                "prerequisites": [],
                "time_limit": None,
                "metadata": {},
            }
            tasks.append(task_data)
            categories.add(task_data["category"])

        return {
            "name": name,
            "description": description,
            "version": "1.0",
            "categories": list(categories),
            "tasks": tasks,
            "progression_rules": {
                "min_success_rate": 0.75,
                "difficulty_increase_threshold": 3,
            },
            "metadata": {"imported_from": "csv"},
        }

    def _import_xml(
        self,
        file_path: Union[Path, BytesIO],
        scenario_name: str = None,
        scenario_description: str = None,
    ) -> Dict[str, Any]:
        """
        Importiert XML-Format
        Erwartet folgende Struktur:
        <scenario>
            <name>...</name>
            <description>...</description>
            <tasks>
                <task>
                    <id>...</id>
                    <question>...</question>
                    <answer>...</answer>
                    ...
                </task>
            </tasks>
        </scenario>
        """
        if isinstance(file_path, BytesIO):
            tree = ET.parse(file_path)
        else:
            tree = ET.parse(file_path)

        root = tree.getroot()

        # Szenario-Metadaten extrahieren
        name = scenario_name or root.findtext(
            "name", f"imported_scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        description = scenario_description or root.findtext(
            "description", "Importiertes Szenario aus XML"
        )

        # Tasks extrahieren
        tasks = []
        categories = set()

        tasks_element = root.find("tasks")
        if tasks_element is not None:
            for task_elem in tasks_element.findall("task"):
                # Hints verarbeiten
                hints = []
                hints_elem = task_elem.find("hints")
                if hints_elem is not None:
                    for hint_elem in hints_elem.findall("hint"):
                        hints.append(hint_elem.text or "")

                task_data = {
                    "id": task_elem.findtext("id", f"task_{len(tasks) + 1:03d}"),
                    "question": task_elem.findtext("question", ""),
                    "answer": task_elem.findtext("answer", ""),
                    "task_type": task_elem.findtext("task_type", "free_text"),
                    "difficulty": int(task_elem.findtext("difficulty", "1")),
                    "category": task_elem.findtext("category", "General"),
                    "hints": hints,
                    "prerequisites": [],
                    "time_limit": None,
                    "metadata": {},
                }
                tasks.append(task_data)
                categories.add(task_data["category"])

        return {
            "name": name,
            "description": description,
            "version": "1.0",
            "categories": list(categories),
            "tasks": tasks,
            "progression_rules": {
                "min_success_rate": 0.75,
                "difficulty_increase_threshold": 3,
            },
            "metadata": {"imported_from": "xml"},
        }

    def _import_excel(
        self,
        file_path: Union[Path, BytesIO],
        scenario_name: str = None,
        scenario_description: str = None,
    ) -> Dict[str, Any]:
        """
        Importiert Excel-Format (XLS/XLSX)
        Ähnlich wie CSV, aber mit Excel-spezifischen Features
        """
        if isinstance(file_path, BytesIO):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_excel(file_path)

        # Standard-Werte setzen
        name = (
            scenario_name
            or f"imported_scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        description = scenario_description or "Importiertes Szenario aus Excel"

        # Tasks konvertieren
        tasks = []
        categories = set()

        for _, row in df.iterrows():
            # Leere Zeilen überspringen
            if pd.isna(row.get("question")) or pd.isna(row.get("answer")):
                continue

            # Hints verarbeiten
            hints = []
            if pd.notna(row.get("hints", "")):
                hints_str = str(row["hints"])
                if ";" in hints_str:
                    hints = [h.strip() for h in hints_str.split(";")]
                elif "," in hints_str:
                    hints = [h.strip() for h in hints_str.split(",")]
                else:
                    hints = [hints_str.strip()]

            task_data = {
                "id": str(row.get("id", f"task_{len(tasks) + 1:03d}")),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "task_type": str(row.get("task_type", "free_text")),
                "difficulty": int(row.get("difficulty", 1)),
                "category": str(row.get("category", "General")),
                "hints": hints,
                "prerequisites": [],
                "time_limit": int(row["time_limit"])
                if pd.notna(row.get("time_limit"))
                else None,
                "metadata": {},
            }
            tasks.append(task_data)
            categories.add(task_data["category"])

        return {
            "name": name,
            "description": description,
            "version": "1.0",
            "categories": list(categories),
            "tasks": tasks,
            "progression_rules": {
                "min_success_rate": 0.75,
                "difficulty_increase_threshold": 3,
            },
            "metadata": {"imported_from": "excel"},
        }

    def _create_scenario_from_data(self, data: Dict[str, Any]) -> LearningScenario:
        """Erstellt LearningScenario-Objekt aus Dictionary"""
        # Tasks konvertieren
        tasks = []
        for task_data in data.get("tasks", []):
            try:
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
            except Exception as e:
                print(
                    f"Fehler beim Erstellen von Task {task_data.get('id', 'unknown')}: {e}"
                )
                continue

        # Szenario erstellen
        scenario = LearningScenario(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0"),
            tasks=tasks,
            categories=data.get("categories", []),
            progression_rules=data.get("progression_rules", {}),
            metadata=data.get("metadata", {}),
        )

        return scenario

    def generate_template(self, format_type: str, output_path: str = None) -> str:
        """
        Generiert Template-Dateien für verschiedene Formate

        Args:
            format_type: 'csv', 'xml', 'xlsx'
            output_path: Ausgabepfad (optional)

        Returns:
            Pfad zur generierten Template-Datei
        """
        if not output_path:
            output_path = f"scenario_template.{format_type}"

        if format_type == "csv":
            return self._generate_csv_template(output_path)
        elif format_type == "xml":
            return self._generate_xml_template(output_path)
        elif format_type == "xlsx":
            return self._generate_xlsx_template(output_path)
        else:
            raise ValueError(f"Template für Format {format_type} nicht unterstützt")

    def _generate_csv_template(self, output_path: str) -> str:
        """Generiert CSV-Template"""
        template_data = [
            [
                "id",
                "question",
                "answer",
                "task_type",
                "difficulty",
                "category",
                "hints",
            ],
            [
                "task_001",
                "Was ist 2+2?",
                "4",
                "free_text",
                "1",
                "Mathematik",
                "Einfache Addition",
            ],
            [
                "task_002",
                "Was ist die Hauptstadt von Deutschland?",
                "Berlin",
                "free_text",
                "2",
                "Geographie",
                "Großstadt;Bundeshauptstadt",
            ],
            [
                "task_003",
                "Übersetze: Hello",
                "Hallo",
                "translation",
                "1",
                "Englisch",
                "Begrüßung",
            ],
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(template_data)

        return output_path

    def _generate_xml_template(self, output_path: str) -> str:
        """Generiert XML-Template"""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<scenario>
    <name>template_scenario</name>
    <description>Template für Szenario-Import</description>
    <version>1.0</version>
    <categories>
        <category>Mathematik</category>
        <category>Geographie</category>
        <category>Englisch</category>
    </categories>
    <tasks>
        <task>
            <id>task_001</id>
            <question>Was ist 2+2?</question>
            <answer>4</answer>
            <task_type>free_text</task_type>
            <difficulty>1</difficulty>
            <category>Mathematik</category>
            <hints>
                <hint>Einfache Addition</hint>
            </hints>
        </task>
        <task>
            <id>task_002</id>
            <question>Was ist die Hauptstadt von Deutschland?</question>
            <answer>Berlin</answer>
            <task_type>free_text</task_type>
            <difficulty>2</difficulty>
            <category>Geographie</category>
            <hints>
                <hint>Großstadt</hint>
                <hint>Bundeshauptstadt</hint>
            </hints>
        </task>
    </tasks>
    <progression_rules>
        <min_success_rate>0.75</min_success_rate>
        <difficulty_increase_threshold>3</difficulty_increase_threshold>
    </progression_rules>
</scenario>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        return output_path

    def _generate_xlsx_template(self, output_path: str) -> str:
        """Generiert Excel-Template"""
        template_data = {
            "id": ["task_001", "task_002", "task_003"],
            "question": [
                "Was ist 2+2?",
                "Was ist die Hauptstadt von Deutschland?",
                "Übersetze: Hello",
            ],
            "answer": ["4", "Berlin", "Hallo"],
            "task_type": ["free_text", "free_text", "translation"],
            "difficulty": [1, 2, 1],
            "category": ["Mathematik", "Geographie", "Englisch"],
            "hints": ["Einfache Addition", "Großstadt;Bundeshauptstadt", "Begrüßung"],
            "time_limit": [30, 60, 15],
        }

        df = pd.DataFrame(template_data)
        df.to_excel(output_path, index=False)

        return output_path


# Helper-Funktionen für Validierung
def validate_imported_scenario(scenario: LearningScenario) -> List[str]:
    """Validiert ein importiertes Szenario und gibt Warnungen zurück"""
    warnings = []

    if not scenario.name:
        warnings.append("Szenario hat keinen Namen")

    if not scenario.tasks:
        warnings.append("Szenario hat keine Tasks")

    for i, task in enumerate(scenario.tasks):
        if not task.question.strip():
            warnings.append(f"Task {task.id}: Leere Frage")

        if not task.answer.strip():
            warnings.append(f"Task {task.id}: Leere Antwort")

        if task.difficulty.value < 1 or task.difficulty.value > 4:
            warnings.append(
                f"Task {task.id}: Ungültiger Schwierigkeitsgrad {task.difficulty.value}"
            )

    return warnings
