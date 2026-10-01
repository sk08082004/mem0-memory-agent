import json
import os


class KnowledgeGraph:
    def __init__(self, file_path="graph.json"):
        self.path_file = file_path
        self.graph = self._load()

    def _load(self):
        if not os.path.exists(self.path_file):
            return {}

        try:
            with open(self.path_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error loading graph from {self.path_file}: {e}")
            return {}

    def save(self):
        with open(self.path_file, "w", encoding="utf-8") as file:
            json.dump(self.graph, file, indent=2)

    def get_user_graph(self, user_id):
        return self.graph.setdefault(
            user_id,
            {
                "nodes": [],
                "relationships": []
            }
        )

    def add_node(self, user_id, memory_id, memory_text):
        user_graph = self.get_user_graph(user_id)

        for node in user_graph["nodes"]:
            if node["id"] == memory_id:
                node["memory_text"] = memory_text
                self.save()
                return

        user_graph["nodes"].append(
            {
                "id": memory_id,
                "memory_text": memory_text
            }
        )

        self.save()

    def add_relationship(
        self,
        user_id,
        source_id,
        target_id,
        relationship
    ):
        user_graph = self.get_user_graph(user_id)

        for edge in user_graph["relationships"]:
            if (
                edge["source"] == source_id
                and edge["target"] == target_id
                and edge["relationship"] == relationship
            ):
                return

        user_graph["relationships"].append(
            {
                "source": source_id,
                "target": target_id,
                "relationship": relationship
            }
        )

        self.save()