import os
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from api.utils.settings import get_setting

from .elevenlabs_phone_call import Task


class StoredTask(BaseModel):
    task_id: UUID
    messages: list
    task: Optional[dict] = None


class MongoDB:
    _instance = None
    _client: Optional[AsyncIOMotorClient] = None
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

    async def store_task(self, task: Task) -> UUID:
        """Store a task in MongoDB with a UUID"""
        await self.connect()

        task_id = uuid4()
        task_data = {
            "task_id": task_id,
            "business_name": task.business_name,
            "business_phone_number": task.business_phone_number,
            "task": task.task,
            "created_at": datetime.utcnow(),
        }

        await self._db.tasks.insert_one(task_data)
        return task_id

    async def get_task(self, task_id: UUID) -> Optional[Task]:
        """Retrieve a task by its UUID"""
        await self.connect()

        task_data = await self._db.tasks.find_one({"task_id": task_id})
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
