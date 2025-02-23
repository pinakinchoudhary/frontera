# workflows.py
from datetime import timedelta
from temporalio import workflow
import activities  # Import activities from activities.py

@workflow.defn
class WavAudioClassificationWorkflow:
    """
    Workflow that accepts the filepath of a wav file, preprocesses it,
    stores it again as a .wav file, and returns the classification results.
    """
    @workflow.run
    async def run(self, wav_filepath: str) -> dict:
        # Preprocess the wav file
        preprocessed_filepath = await workflow.execute_activity(
            activities.preprocess_audio,
            wav_filepath,
            start_to_close_timeout=timedelta(seconds=30)
        )
        classification = await workflow.execute_activity(
            activities.classify_audio,
            preprocessed_filepath,
            start_to_close_timeout=timedelta(seconds=60)
        )
        # Store the classification results
        await workflow.execute_activity(
            activities.store_results,
            classification,
            start_to_close_timeout=timedelta(seconds=30)
        )
        return classification
