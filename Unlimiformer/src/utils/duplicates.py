# Drop duplicates function
def drop_duplicates_in_input(untokenized_dataset):
    seen_documents = set()
    indices_to_keep = []
    for i, document in enumerate(untokenized_dataset["document"]):
        if document not in seen_documents:
            seen_documents.add(document)
            indices_to_keep.append(i)
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    return untokenized_dataset
