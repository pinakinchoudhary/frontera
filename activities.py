import asyncio
from temporalio import activity
from ensemble_classify import preprocess_and_save_wav, classify_wrapper

@activity.defn
async def preprocess_audio(filepath) -> str:
    """
    Receives raw audio data and performs necessary preprocessing steps,
    such as normalization, noise reduction, and feature extraction.
    """
    processed_filepath = preprocess_and_save_wav(filepath)
    print("Preprocessing complete. File saved at:", processed_filepath)
    return processed_filepath

@activity.defn
async def classify_audio(processed_filepath) -> dict:
    """
    Runs the ensemble model on the preprocessed audio data to classify it.
    The ensemble could combine predictions from multiple models.
    """
    classification_result = classify_wrapper(processed_filepath)
    print("Classification complete. Result:", classification_result)
    return classification_result

@activity.defn
async def store_results(results: dict) -> None:
    """
    Stores the classification results in your chosen persistence layer,
    such as a database or a cloud storage service.
    """
    # Store results in a local json file
    with open("results.json", "a") as f:
        f.write(str(results) + "\n")
    print("Storing results:", results)
    return