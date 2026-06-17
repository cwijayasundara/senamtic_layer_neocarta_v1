import sqlite3

from data.seed_sqlite import seed_all


def test_seed_creates_both_databases(tmp_path):
    paths = seed_all(out_dir=str(tmp_path))
    assert (tmp_path / "financials.db").exists()
    assert (tmp_path / "org.db").exists()
    assert set(paths) == {"financials", "org"}


def test_financials_tables_populated(tmp_path):
    seed_all(out_dir=str(tmp_path))
    con = sqlite3.connect(tmp_path / "financials.db")
    assert con.execute("SELECT COUNT(*) FROM income_statement").fetchone()[0] == 13
    assert con.execute("SELECT COUNT(*) FROM stock_price").fetchone()[0] > 0
    con.close()


def test_org_join_returns_headcount_by_region(tmp_path):
    seed_all(out_dir=str(tmp_path))
    con = sqlite3.connect(tmp_path / "org.db")
    rows = con.execute(
        """
        SELECT l.region, SUM(h.employee_count)
        FROM headcount h
        JOIN location l ON l.location_id = h.location_id
        JOIN department d ON d.department_id = h.department_id
        WHERE h.fiscal_year = 2025 AND h.quarter = 'Q4'
        GROUP BY l.region
        """
    ).fetchall()
    con.close()
    assert len(rows) > 0
    assert all(total > 0 for _, total in rows)


def test_org_db_enforces_headcount_foreign_keys(tmp_path):
    import pytest

    seed_all(out_dir=str(tmp_path))
    con = sqlite3.connect(tmp_path / "org.db")
    con.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO headcount "
            "(snapshot_id, department_id, location_id, fiscal_year, quarter, employee_count) "
            "VALUES (999999, 12345, 1, 2025, 'Q4', 10)"
        )
    con.close()


def test_seed_all_is_rerunnable_into_existing_dir(tmp_path):
    # Re-seeding an existing org.db must not fail FK enforcement on table drops.
    seed_all(out_dir=str(tmp_path))
    seed_all(out_dir=str(tmp_path))  # would raise sqlite3.IntegrityError before the fix
    con = sqlite3.connect(tmp_path / "org.db")
    assert con.execute("SELECT COUNT(*) FROM headcount").fetchone()[0] == 240
    con.close()
