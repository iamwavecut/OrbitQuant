from orbitquant.eval.prompts import build_prompt_seed_jobs, default_prompt_payload


def test_build_prompt_seed_jobs_crosses_selected_prompts_and_seeds():
    payload = default_prompt_payload("flux2")

    jobs = build_prompt_seed_jobs(
        payload,
        seeds=[0, 1],
        prompt_ids=["simple-object", "counting"],
    )

    assert [
        (job["prompt_record"]["id"], job["seed"])
        for job in jobs
    ] == [
        ("simple-object", 0),
        ("counting", 0),
        ("simple-object", 1),
        ("counting", 1),
    ]


def test_build_prompt_seed_jobs_applies_prompt_limit_after_id_filter():
    payload = default_prompt_payload("flux2")

    jobs = build_prompt_seed_jobs(
        payload,
        seeds=[5],
        prompt_ids=["simple-object", "counting"],
        prompt_limit=1,
    )

    assert len(jobs) == 1
    assert jobs[0]["prompt_record"]["id"] == "simple-object"
    assert jobs[0]["seed"] == 5
