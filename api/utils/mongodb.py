from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from api.utils.settings import get_setting

from .elevenlabs_phone_call import Task


class TaskStatus(str, Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskUpdate(BaseModel):
    task_id: str  # This will store the string representation of ObjectId
    status: TaskStatus
    message: str
    timestamp: datetime


class StoredTask(BaseModel):
    task_id: str  # This will store the string representation of ObjectId
    messages: list
    task: Optional[dict] = None


class MongoDB:
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
        return cls._instance

    async def connect(self):
        if not self._client:
            mongodb_uri = get_setting("MONGODB_URI")
            self._client = AsyncIOMotorClient(mongodb_uri)
            self._db = self._client.bubbacall

    async def store_task(self, task: Task) -> str:
        """Store a task in MongoDB and return the ObjectId as string"""
        await self.connect()

        task_data = {
            "business_name": task.business_name,
            "business_phone_number": task.business_phone_number,
            "task": task.task,
            "status": TaskStatus.CREATED,
            "created_at": datetime.now(),
            "updates": [
                {
                    "status": TaskStatus.CREATED,
                    "message": "Task created",
                    "timestamp": datetime.now(),
                }
            ],
        }

        result = await self._db.tasks.insert_one(task_data)
        return str(result.inserted_id)

    async def update_task_status(self, task_id: str, status: TaskStatus, message: str):
        """Update task status and add a new update entry"""
        await self.connect()

        update = {"status": status, "message": message, "timestamp": datetime.now()}

        await self._db.tasks.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {"status": status}, "$push": {"updates": update}},
        )

    async def watch_task_updates(
        self, task_id: str
    ) -> AsyncGenerator[TaskUpdate, None]:
        """Watch for updates to a specific task"""
        await self.connect()

        # Get initial task state
        task = await self._db.tasks.find_one({"_id": ObjectId(task_id)})
        if task and task.get("updates"):
            for update in task["updates"]:
                yield TaskUpdate(
                    task_id=task_id,
                    status=update["status"],
                    message=update["message"],
                    timestamp=update["timestamp"],
                )

        # Watch for new updates
        pipeline = [
            {
                "$match": {
                    "operationType": "update",
                    "documentKey._id": ObjectId(task_id),
                }
            }
        ]

        async with self._db.tasks.watch(pipeline) as change_stream:
            async for change in change_stream:
                # The updates look like updates.0, updates.1, etc, so this handles that.
                # TODO(ege): This doesn't handle a case where a single update call
                # adds multiple messages.
                for k, v in change["updateDescription"]["updatedFields"].items():
                    if not k.startswith("updates."):
                        continue
                    yield TaskUpdate(
                        task_id=task_id,
                        status=v["status"],
                        message=v["message"],
                        timestamp=v["timestamp"],
                    )

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by its ObjectId"""
        await self.connect()

        task_data = await self._db.tasks.find_one({"_id": ObjectId(task_id)})
        if task_data:
            return Task(
                business_name=task_data["business_name"],
                business_phone_number=task_data["business_phone_number"],
                task=task_data["task"],
            )
        return None

    async def close(self):
        """Close the MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
