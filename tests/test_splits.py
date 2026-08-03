from tools.create_splits import assign_groups


def test_sequences_never_cross_splits_and_assignment_is_deterministic():
    images = []
    image_id = 1
    for source in ("smd", "seadronessee", "ships_vessels"):
        for sequence in range(10):
            for _ in range(sequence % 3 + 1):
                images.append(
                    {
                        "id": image_id,
                        "source": source,
                        "sequence_id": f"{source}_{sequence}",
                    }
                )
                image_id += 1
    first = assign_groups(images, (0.7, 0.15, 0.15), 2026)
    second = assign_groups(images, (0.7, 0.15, 0.15), 2026)
    assert first == second
    membership = {
        image_id: split for split, image_ids in first.items() for image_id in image_ids
    }
    by_sequence = {}
    for image in images:
        split = membership[image["id"]]
        key = (image["source"], image["sequence_id"])
        by_sequence.setdefault(key, split)
        assert by_sequence[key] == split
    assert set(membership) == {image["id"] for image in images}
