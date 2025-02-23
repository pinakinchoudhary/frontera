# worker.py
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
import workflows
import activities

async def main():
    # Connect to the Temporal server
    client = await Client.connect("localhost:7233")
    
    # Create a worker for our workflows and activities
    worker = Worker(
        client,
        task_queue="audio-classification-task-queue",
        workflows=[workflows.WavAudioClassificationWorkflow],
        activities=[activities.preprocess_audio, activities.classify_audio, activities.store_results],
    )
    
    print("Worker started. Listening for tasks...")
    # Run worker until terminated.
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
