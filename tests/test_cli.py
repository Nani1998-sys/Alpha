"""Command-line interface: commands, option placement and error messages."""
import json
import os

from alpha.cli import main


def test_db_option_after_subcommand(tmp_path, capsys):
    """`alpha add --db x.json ...` used to fail with 'unrecognized arguments'."""
    db = os.path.join(tmp_path, "d.json")
    assert main(["add", "--db", db, "--name", "X", "--smiles", "CCO",
                 "--cl", "1", "--vc", "5"]) == 0
    assert main(["list", "--db", db]) == 0
    assert "X" in capsys.readouterr().out
    assert json.load(open(db, encoding="utf-8"))["drugs"][0]["name"] == "X"


def test_db_option_before_subcommand(tmp_path):
    db = os.path.join(tmp_path, "d.json")
    assert main(["--db", db, "add", "--name", "Y", "--cl", "1", "--vc", "5"]) == 0
    assert main(["--db", db, "simulate", "--name", "Y", "--dose", "5",
                 "--route", "iv_bolus"]) == 0


def test_unknown_drug_message_is_clean(tmp_path, capsys):
    db = os.path.join(tmp_path, "d.json")
    assert main(["show", "--db", db, "--name", "Nope"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: No drug named 'Nope'")  # no KeyError quoting


def test_simulate_writes_outputs(tmp_path):
    db = os.path.join(tmp_path, "d.json")
    main(["add", "--db", db, "--name", "Z", "--smiles", "CCO", "--cl", "2", "--vc", "10",
          "--ka", "1", "--pd", "emax", "--emax", "100", "--ec50", "1"])
    csv_p, json_p = os.path.join(tmp_path, "o.csv"), os.path.join(tmp_path, "o.json")
    assert main(["simulate", "--db", db, "--name", "Z", "--dose", "50", "--interval", "8",
                 "--ndoses", "3", "--csv", csv_p, "--json", json_p]) == 0
    assert open(csv_p, encoding="utf-8").readline().startswith("time,concentration,effect")
    assert json.load(open(json_p, encoding="utf-8"))["metrics"]["Cmax"] > 0


def test_invalid_structure_is_reported(tmp_path, capsys):
    db = os.path.join(tmp_path, "d.json")
    assert main(["add", "--db", db, "--name", "Bad", "--smiles", "C1CC",
                 "--cl", "1", "--vc", "1"]) == 1
    assert "SMILES" in capsys.readouterr().err
