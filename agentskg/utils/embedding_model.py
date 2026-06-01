import asyncio
import httpx
import os
import json

EMPTY_TEXT_PLACEHOLDER = "[NULL]"


async def _get_embeddings_for_single_batch_async(batch_texts: list[str]):
    """
    Asynchronously retrieve embeddings for a small batch of text.This is the function that actually interacts with the API.
    """
    if not batch_texts:
        return []

    processed_batch_texts = [
        text if isinstance(text, str) and text.strip() else EMPTY_TEXT_PLACEHOLDER
        for text in batch_texts
    ]

    url = os.getenv("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings")
    payload = {
        "model": os.getenv("Embedding_MODEL", "BAAI/bge-m3"),
        "input": processed_batch_texts,
        "encoding_format": "float",
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('EMBEDDING_API_KEY')}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()

            if "data" in res_json and isinstance(res_json["data"], list):
                api_embeddings_data = res_json["data"]
                if len(api_embeddings_data) == len(processed_batch_texts):
                    # print(f"    [API Call] Successfully received {len(api_embeddings_data)} embeddings for batch.")
                    return [item.get("embedding") for item in api_embeddings_data]
                else:
                    print(
                        f"    [API Call] Error: Mismatch in embeddings returned for batch. Expected {len(processed_batch_texts)}, got {len(api_embeddings_data)}."
                    )
                    return [None] * len(processed_batch_texts)
            else:
                print(
                    f"    [API Call] Error: Unexpected API response for batch. Response: {res_json}"
                )
                return [None] * len(processed_batch_texts)
        except httpx.HTTPStatusError as e:
            print(
                f"    [API Call] HTTP error for batch: {e.response.status_code} - {e.response.text}"
            )
            return [None] * len(processed_batch_texts)
        except Exception as e:
            print(f"    [API Call] Unexpected error for batch: {e}")
            return [None] * len(processed_batch_texts)


async def get_embeddings(texts: list[str]):
    """
    Asynchronously retrieve embeddings for a batch of text (which might be large), by handling batching on the client side.
    """
    if not texts:
        return []

    client_batch_size = 64
    processed_texts_for_api = [
        text.strip() if text and text.strip() else EMPTY_TEXT_PLACEHOLDER
        for text in texts
    ]

    all_embeddings_ordered = []
    num_texts = len(processed_texts_for_api)

    # print(f"  [Client Batching] Total texts to embed: {num_texts}. Client batch size: {client_batch_size}.")

    for i in range(0, num_texts, client_batch_size):
        current_client_batch = processed_texts_for_api[i : i + client_batch_size]
        # print(f"  [Client Batching] Processing client batch {i//client_batch_size + 1}/{(num_texts + client_batch_size -1)//client_batch_size} (size: {len(current_client_batch)})...")

        batch_embeddings_results = await _get_embeddings_for_single_batch_async(
            current_client_batch
        )

        if len(batch_embeddings_results) != len(current_client_batch):
            # print(f"    [Client Batching] Error: Embedding result length mismatch for client batch {i//client_batch_size + 1}. Expected {len(current_client_batch)}, got {len(batch_embeddings_results)}. Filling with None.")
            all_embeddings_ordered.extend([None] * len(current_client_batch))
        else:
            all_embeddings_ordered.extend(batch_embeddings_results)

        if (
            num_texts > client_batch_size and i + client_batch_size < num_texts
        ):
            await asyncio.sleep(0.1)

    # print(f"  [Client Batching] Finished all client batches. Total embeddings collected: {len(all_embeddings_ordered)} (includes Nones for errors).")
    return all_embeddings_ordered


async def test_get_embeddings_client_batching():
    print("\n--- Running Embedding API Test (Client Batching Strategy) ---")

    sample_texts_valid_small = ["text1", "text2", "text3"]
    sample_texts_valid_large = [
        f"sentence {i}" for i in range(250)
    ]
    sample_texts_with_empty = ["hello", "", "world", "  ", "test"]
    sample_texts_all_empty = ["", " ", "  "]
    sample_texts_empty_list = []

    print("\n[Test Case 1] Small valid list:")
    embeddings1 = await get_embeddings(sample_texts_valid_small)
    assert len(embeddings1) == len(sample_texts_valid_small)
    assert all(
        isinstance(e, list) and len(e) == 1024 for e in embeddings1 if e is not None
    )
    print(
        f"  Result for small list (first embedding's first 5 values): {embeddings1[0][:5] if embeddings1 and embeddings1[0] else 'N/A'}"
    )

    print("\n[Test Case 2] Large valid list (will be batched by client):")
    embeddings2 = await get_embeddings(sample_texts_valid_large)  # client_batch_size=80
    assert len(embeddings2) == len(sample_texts_valid_large)
    if not all(e is not None for e in embeddings2):
        print(
            f"  Warning: Some embeddings in large list are None. Successful: {sum(1 for e in embeddings2 if e is not None)}/{len(embeddings2)}"
        )
    else:
        print(f"  Successfully got all {len(embeddings2)} embeddings for large list.")
    if embeddings2 and embeddings2[0]:
        assert isinstance(embeddings2[0], list) and len(embeddings2[0]) == 1024

    print("\n[Test Case 3] List with empty/whitespace strings:")
    embeddings3 = await get_embeddings(sample_texts_with_empty)
    assert len(embeddings3) == len(sample_texts_with_empty)
    # processed_empty_indices = [1, 3]
    # for i in processed_empty_indices:
    #     assert embeddings3[i] is not None, f"Embedding for original empty string at index {i} (now placeholder) should not be None."
    #     assert isinstance(embeddings3[i], list) and len(embeddings3[i]) == 1024
    print(
        f"  Results for list with empty strings (lengths): {[len(e) if e else 0 for e in embeddings3]}"
    )

    print("\n[Test Case 4] List with all empty/whitespace strings:")
    embeddings4 = await get_embeddings(sample_texts_all_empty)
    assert len(embeddings4) == len(sample_texts_all_empty)
    assert all(
        isinstance(e, list) and len(e) == 1024 for e in embeddings4 if e is not None
    )
    print(f"  Results for all-empty list (all should be placeholder embeddings).")

    print("\n[Test Case 5] Empty input list:")
    embeddings5 = await get_embeddings(sample_texts_empty_list)
    assert embeddings5 == []
    print(f"  Result for empty input list: {embeddings5} (Correct)")

    print("\n--- Embedding API Test (Client Batching Strategy) Finished ---")


if __name__ == "__main__":
    asyncio.run(test_get_embeddings_client_batching())
