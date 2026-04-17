# run_worker.py
from redis import Redis
from rq import Worker, Queue, Connection
import os

listen = ["default"]
redis_url = os.getenv("REDIS_URL", f"redis://{os.getenv('REDIS_HOST','localhost')}:{os.getenv('REDIS_PORT','6379')}")

conn = Redis.from_url(redis_url)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
