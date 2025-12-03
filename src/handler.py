import runpod
import aiohttp
import asyncio
from typing import Dict, Any

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

# Configuration for aiohttp session
TIMEOUT = aiohttp.ClientTimeout(total=600)
RETRY_ATTEMPTS = 10
RETRY_DELAY = 0.1


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
async def wait_for_service(url: str) -> None:
    """
    Check if the service is ready to receive requests.
    """
    retries = 0
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                retries += 1

                # Only log every 15 retries so the logs don't get spammed
                if retries % 15 == 0:
                    print("Service not ready yet. Retrying...")
            except Exception as err:
                print("Error: ", err)

            await asyncio.sleep(0.2)


async def run_inference(inference_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on a request with retry logic.
    """
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with session.post(
                    url=f'{LOCAL_URL}/txt2img',
                    json=inference_request
                ) as response:
                    if response.status in [502, 503, 504] and attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                raise Exception(f"Failed to run inference after {RETRY_ATTEMPTS} attempts: {str(e)}")
        
        raise Exception(f"Failed to run inference after {RETRY_ATTEMPTS} attempts")


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is the async handler function that will be called by the serverless.
    Processes requests concurrently for better resource utilization.
    
    Args:
        job: Contains the input data and request metadata
        
    Returns:
        Dict containing the generated image data
    """
    try:
        job_input = job["input"]
        result = await run_inference(job_input)
        
        # return the output that you want to be returned like pre-signed URLs to output artifacts
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def concurrency_modifier(current_concurrency: int) -> int:
    """
    Dynamically adjust the worker's concurrency level.
    
    Args:
        current_concurrency: The current concurrency level
        
    Returns:
        int: The new concurrency level
    """
    # Configuration for concurrency limits
    # Adjust these based on your GPU memory and performance testing
    max_concurrency = 5  # Maximum concurrent requests
    min_concurrency = 1  # Minimum concurrency to maintain
    
    # For image generation workloads, we typically want to maintain
    # a stable concurrency level based on GPU capacity
    # Start with conservative settings and adjust based on monitoring
    
    # You can implement dynamic scaling logic here based on:
    # - GPU memory usage
    # - Average processing time
    # - Queue depth
    # For now, we'll use a fixed optimal concurrency
    
    optimal_concurrency = 3  # Good balance for most GPU setups
    return min(optimal_concurrency, max_concurrency)


if __name__ == "__main__":
    # Wait for service using asyncio
    asyncio.run(wait_for_service(url=f'{LOCAL_URL}/sd-models'))
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    print(f"Concurrent request handling enabled (max concurrency: 5)")
    
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier
    })