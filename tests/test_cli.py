from orbitquant.cli.main import main


def test_cli_version_prints_version(capsys):
    assert main(["--version"]) == 0

    output = capsys.readouterr().out
    assert "0.1.0" in output


def test_cli_native_suites_lists_no_range_smoke_settings(capsys):
    assert main(["native-suites"]) == 0

    output = capsys.readouterr().out
    assert "flux2-native" in output
    assert "wan-native" in output
    assert "range" not in output.lower()


def test_cli_generate_requires_prompt_and_output():
    try:
        main(["generate", "--suite", "flux2-native"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("generate accepted missing prompt/output arguments")


def test_cli_generate_dry_run_prints_native_request(capsys, tmp_path):
    assert (
        main(
            [
                "generate",
                "--suite",
                "flux2-native",
                "--prompt",
                "A native prompt",
                "--output",
                str(tmp_path),
                "--seed",
                "9",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "black-forest-labs/FLUX.2-klein-4B" in output
    assert '"height": 1024' in output
    assert '"width": 1024' in output
