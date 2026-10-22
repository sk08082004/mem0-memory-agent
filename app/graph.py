import copy
import json
import os
import tempfile
import threading


class KnowledgeGraph:
    def __init__(self, file_path="graph.json"):
        self.path_file = file_path
        self._lock = threading.RLock()
        self.graph = self._load()

    def _load(self):
        if not os.path.exists(self.path_file):
            return {}

        try:
            with open(self.path_file, "r", encoding="utf-8") as file:
                return json.load(file)

        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Error loading graph from {self.path_file}: {e}"
            )
            return {}

    def save(self, graph=None):
        with self._lock:
            graph_to_save = self.graph if graph is None else graph
            directory = os.path.dirname(os.path.abspath(self.path_file))
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".graph-",
                suffix=".tmp",
                dir=directory
            )
            try:
                with os.fdopen(
                    descriptor,
                    "w",
                    encoding="utf-8"
                ) as file:
                    json.dump(
                        graph_to_save,
                        file,
                        indent=2
                    )
                os.replace(temporary_path, self.path_file)
            except Exception:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                raise

    def _commit_user_graph(self, user_id, user_graph):
        updated_graph = dict(self.graph)
        updated_graph[user_id] = user_graph
        self.save(updated_graph)
        self.graph = updated_graph

    def get_user_graph(self, user_id):
        return self.graph.setdefault(
            user_id,
            {
                "nodes": [],
                "relationships": []
            }
        )

    def add_node(
        self,
        user_id,
        memory_id,
        memory_text
    ):
        with self._lock:
            user_graph = copy.deepcopy(
                self.graph.get(
                    user_id,
                    {
                        "nodes": [],
                        "relationships": []
                    }
                )
            )

            for node in user_graph["nodes"]:
                if node["id"] == memory_id:
                    node["memory_text"] = memory_text
                    self._commit_user_graph(user_id, user_graph)
                    return

            user_graph["nodes"].append(
                {
                    "id": memory_id,
                    "memory_text": memory_text
                }
            )

            self._commit_user_graph(user_id, user_graph)

    def add_relationship(
        self,
        user_id,
        source_id,
        target_id,
        relationship
    ):
        """Add an edge if both endpoints exist; return False when rejected."""

        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(target_id, str)
            or not target_id.strip()
            or not isinstance(relationship, str)
            or not relationship.strip()
        ):
            return False

        with self._lock:
            user_graph = self.graph.get(user_id)
            if (
                not isinstance(user_graph, dict)
                or not isinstance(user_graph.get("nodes"), list)
                or not isinstance(user_graph.get("relationships"), list)
            ):
                return False

            node_ids = {
                node.get("id")
                for node in user_graph["nodes"]
                if (
                    isinstance(node, dict)
                    and isinstance(node.get("id"), str)
                    and node["id"].strip()
                )
            }
            if source_id not in node_ids or target_id not in node_ids:
                return False

            for edge in user_graph["relationships"]:
                if (
                    isinstance(edge, dict)
                    and edge.get("source") == source_id
                    and edge.get("target") == target_id
                    and edge.get("relationship") == relationship
                ):
                    return True

            updated_user_graph = copy.deepcopy(user_graph)
            updated_user_graph["relationships"].append(
                {
                    "source": source_id,
                    "target": target_id,
                    "relationship": relationship
                }
            )

            self._commit_user_graph(user_id, updated_user_graph)
            return True

    def delete_node(
        self,
        user_id,
        memory_id
    ):
        with self._lock:
            user_graph = copy.deepcopy(
                self.graph.get(
                    user_id,
                    {
                        "nodes": [],
                        "relationships": []
                    }
                )
            )

            user_graph["nodes"] = [
                node
                for node in user_graph["nodes"]
                if node["id"] != memory_id
            ]

            user_graph["relationships"] = [
                relationship
                for relationship in user_graph["relationships"]
                if (
                    relationship["source"] != memory_id
                    and relationship["target"] != memory_id
                )
            ]

            self._commit_user_graph(user_id, user_graph)

    def sync_user_graph(
        self,
        user_id,
        memories
    ):
        """
        Synchronize the user's knowledge graph with
        the currently existing Mem0 memories.
        """

        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(memories, dict)
            or not isinstance(memories.get("results"), list)
            or type(memories.get("count")) is not int
            or memories["count"] != len(memories["results"])
        ):
            raise ValueError(
                "Cannot synchronize graph from an incomplete or invalid memory listing."
            )

        current_memories = {}

        for memory in memories.get("results", []):
            if (
                not isinstance(memory, dict)
                or not isinstance(memory.get("id"), str)
                or not memory["id"].strip()
                or not isinstance(memory.get("memory"), str)
                or not memory["memory"].strip()
            ):
                raise ValueError(
                    "Cannot synchronize graph from a memory without a valid ID and usable text."
                )

            memory_id = memory["id"]
            if memory_id in current_memories:
                raise ValueError(
                    "Cannot synchronize graph from duplicate memory IDs."
                )
            current_memories[memory_id] = memory["memory"].strip()

        valid_ids = set(
            current_memories.keys()
        )

        with self._lock:
            user_graph = self.graph.get(user_id)
            if user_graph is None:
                user_graph = {
                    "nodes": [],
                    "relationships": []
                }
            elif (
                not isinstance(user_graph, dict)
                or not isinstance(user_graph.get("relationships"), list)
            ):
                raise ValueError(
                    "Cannot synchronize an invalid existing graph."
                )

            valid_relationships = [
                relationship
                for relationship in user_graph["relationships"]
                if (
                    isinstance(relationship, dict)
                    and isinstance(relationship.get("source"), str)
                    and relationship["source"] in valid_ids
                    and isinstance(relationship.get("target"), str)
                    and relationship["target"] in valid_ids
                )
            ]
            current_nodes = [
                {
                    "id": memory_id,
                    "memory_text": memory_text
                }
                for memory_id, memory_text
                in current_memories.items()
            ]

            updated_user_graph = copy.deepcopy(user_graph)
            updated_user_graph["nodes"] = current_nodes
            updated_user_graph["relationships"] = valid_relationships

            self._commit_user_graph(user_id, updated_user_graph)