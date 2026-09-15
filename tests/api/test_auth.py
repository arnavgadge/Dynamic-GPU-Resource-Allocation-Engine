import pytest

from api import auth


def test_login_returns_the_seeded_account():
    account = auth.login("user_a")
    assert account.username == "user_a"
    assert account.role == "USER"
    assert account.display_name == "User A"


def test_admin_account_has_admin_role():
    account = auth.login("admin")
    assert account.role == "ADMIN"


def test_login_rejects_unknown_username():
    with pytest.raises(KeyError):
        auth.login("not_a_real_user")


def test_create_token_and_resolve_round_trip():
    token = auth.create_token("user_b")
    account = auth.resolve_token(token)
    assert account is not None
    assert account.username == "user_b"


def test_resolve_unknown_token_returns_none():
    assert auth.resolve_token("not-a-real-token") is None


def test_logout_invalidates_the_token():
    token = auth.create_token("user_c")
    assert auth.resolve_token(token) is not None
    auth.logout(token)
    assert auth.resolve_token(token) is None


def test_each_login_gets_a_distinct_token():
    token1 = auth.create_token("user_a")
    token2 = auth.create_token("user_a")
    assert token1 != token2
