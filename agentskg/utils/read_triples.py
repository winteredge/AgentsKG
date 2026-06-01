def read_triples_in_batches(file_path, batch_size=1):
    batch = []
    count = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("!"):
                triple = line.split("\t")
                if len(triple) == 3:
                    batch.append(triple)
                    count += 1
                    if count == batch_size:
                        yield batch
                        batch = []
                        count = 0

        if batch:
            yield batch


# file_path = 'agentskg/resources/baike_triples.txt'
# for i, batch in enumerate(read_triples_in_batches(file_path, batch_size=1000)):
#     print(f"Batch {i+1}:")
#     print(batch)


def read_all_triples(file_path):
    triples = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                triple = line.split("\t")
                if len(triple) == 3:
                    triples.append(triple)

    return triples
