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
    assert con.execute("SELECT COUNT(*) FROM income_statement").fetchone()[0] == 8
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
