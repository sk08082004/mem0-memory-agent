import copy
import json
import os
import tempfile
import threading
from pathlib import Path

from app.conversation import normalize_user_conversations


DEFAULT_CONVERSATION_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "conversations.json"
)


class ConversationStoreError(RuntimeError):
    """Raised when local conversation data cannot be safely read or written."""


class JsonConversationRepository:
    """Local, user-scoped transcript storage behind a replaceable interface."""

    VERSION = 1
    _process_lock = threading.RLock()

    def __init__(self, file_path=DEFAULT_CONVERSATION_FILE):
        self.path = Path(file_path).expanduser().resolve()

    def load_user(self, user_id):
        user_id = _validate_user_id(user_id)
        with self._locked():
            document = self._read_document()
        user_data, malformed = normalize_user_conversations(
            document["users"].get(user_id),
            user_id,
            recover_pending=True,
        )
        return user_data, malformed

    def save_user(self, user_id, user_data):
        user_id = _validate_user_id(user_id)
        normalized, malformed = normalize_user_conversations(
            user_data,
            user_id,
        )
        if malformed:
            raise ConversationStoreError(
                "Conversation data failed validation and was not saved."
            )

        with self._locked():
            document = self._read_document()
            document["users"][user_id] = normalized
            self._write_document(document)
        return copy.deepcopy(normalized)

    def create_conversation(self, user_id):
        from app.conversation import start_new_conversation

        created = {}

        def create(user_data):
            created["conversation"] = start_new_conversation(
                user_data,
                user_id,
            )

        user_data = self.update_user(user_id, create)
        conversation = next(
            item
            for item in user_data["conversations"]
            if item["id"] == created["conversation"]["id"]
        )
        return copy.deepcopy(conversation), user_data

    def save_conversation(self, user_id, conversation):
        user_id = _validate_user_id(user_id)
        if (
            not isinstance(conversation, dict)
            or conversation.get("owner_id") != user_id
            or not isinstance(conversation.get("id"), str)
        ):
            raise ValueError("Conversation does not belong to the selected user.")

        with self._locked():
            document = self._read_document()
            user_data, malformed = normalize_user_conversations(
                document["users"].get(user_id),
                user_id,
            )
            if malformed:
                raise ConversationStoreError(
                    "Conversation data failed validation and cannot be updated."
                )
            replacement, replacement_malformed = normalize_user_conversations(
                {
                    "active_id": conversation["id"],
                    "conversations": [conversation],
                },
                user_id,
            )
            if replacement_malformed or not replacement["conversations"]:
                raise ConversationStoreError(
                    "Conversation data failed validation and was not saved."
                )
            replacement = replacement["conversations"][0]
            existing_index = next(
                (
                    index
                    for index, item in enumerate(user_data["conversations"])
                    if item["id"] == replacement["id"]
                ),
                None,
            )
            if existing_index is None:
                user_data["conversations"].append(replacement)
            else:
                user_data["conversations"][existing_index] = replacement
            user_data["active_id"] = replacement["id"]
            document["users"][user_id] = user_data
            self._write_document(document)
        return copy.deepcopy(replacement)

    def set_active_conversation(self, user_id, conversation_id):
        def set_active(user_data):
            if not any(
                item["id"] == conversation_id
                for item in user_data["conversations"]
            ):
                raise ValueError("Conversation was not found for this user.")
            user_data["active_id"] = conversation_id

        return self.update_user(user_id, set_active)

    def rename_conversation(self, user_id, conversation_id, title):
        from app.conversation import rename_conversation

        def rename(user_data):
            conversation = _find_conversation(
                user_data,
                conversation_id,
            )
            rename_conversation(conversation, title)

        return self.update_user(user_id, rename)

    def delete_conversation(self, user_id, conversation_id):
        def delete(user_data):
            index = next(
                (
                    index
                    for index, item in enumerate(user_data["conversations"])
                    if item["id"] == conversation_id
                ),
                None,
            )
            if index is None:
                raise ValueError("Conversation was not found for this user.")
            user_data["conversations"].pop(index)
            user_data["active_id"] = (
                user_data["conversations"][-1]["id"]
                if user_data["conversations"]
                else None
            )

        return self.update_user(user_id, delete)

    def update_user(self, user_id, update):
        user_id = _validate_user_id(user_id)
        with self._locked():
            document = self._read_document()
            user_data, malformed = normalize_user_conversations(
                document["users"].get(user_id),
                user_id,
            )
            if malformed:
                raise ConversationStoreError(
                    "Conversation data failed validation and cannot be updated."
                )
            update(user_data)
            document["users"][user_id] = user_data
            self._write_document(document)
        return copy.deepcopy(user_data)

    def _read_document(self):
        if not self.path.exists():
            return {"version": self.VERSION, "users": {}}
        try:
            with self.path.open("r", encoding="utf-8") as file:
                document = json.load(file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConversationStoreError(
                "Local conversation history could not be read. "
                "The original file was left untouched."
            ) from error

        if (
            not isinstance(document, dict)
            or document.get("version") != self.VERSION
            or not isinstance(document.get("users"), dict)
        ):
            raise ConversationStoreError(
                "Local conversation history has an unsupported or invalid format."
            )
        return document

    def _write_document(self, document):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".conversations-",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(document, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as error:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise ConversationStoreError(
                "Local conversation history could not be saved."
            ) from error

    def _locked(self):
        return _conversation_file_lock(self.path)


def _validate_user_id(user_id):
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("User ID must be a non-empty string.")
    return user_id.strip()


def _find_conversation(user_data, conversation_id):
    conversation = next(
        (
            item
            for item in user_data["conversations"]
            if item["id"] == conversation_id
        ),
        None,
    )
    if conversation is None:
        raise ValueError("Conversation was not found for this user.")
    return conversation


class _conversation_file_lock:
    def __init__(self, path):
        self.path = Path(str(path) + ".lock")
        self.file = None

    def __enter__(self):
        JsonConversationRepository._process_lock.acquire()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a+b")
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b"\0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
            return self
        except OSError as error:
            if self.file is not None:
                self.file.close()
            JsonConversationRepository._process_lock.release()
            raise ConversationStoreError(
                "Local conversation history is locked or unavailable."
            ) from error

    def __exit__(self, error_type, error, traceback):
        try:
            if self.file is not None:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
                self.file.close()
        except OSError as lock_error:
            raise ConversationStoreError(
                "Local conversation history lock could not be released."
            ) from lock_error
        finally:
            JsonConversationRepository._process_lock.release()
        return False
