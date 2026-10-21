from unittest.mock import Mock

import boto3
import pytest

from skillclaw.object_store import S3ObjectStore


@pytest.mark.parametrize("path", ["/my-bucket", "/my-bucket/", "/prefix/data"])
def test_r2_rejects_bucket_suffix_before_opening_session(monkeypatch, path):
    session = Mock()
    monkeypatch.setattr(boto3.session, "Session", session)
    with pytest.raises(ValueError, match="without a bucket/path suffix"):
        S3ObjectStore(
            endpoint="https://account.r2.cloudflarestorage.com" + path,
            bucket="my-bucket",
            access_key_id="",
            secret_access_key="",
        )
    session.assert_not_called()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://account.r2.cloudflarestorage.com",
        "https://account.eu.r2.cloudflarestorage.com/",
        "https://s3.example.test/proxy",
        "",
    ],
)
def test_valid_endpoints_preserve_bucket_and_url(monkeypatch, endpoint):
    client = Mock()
    session = Mock(return_value=Mock(client=Mock(return_value=client)))
    monkeypatch.setattr(boto3.session, "Session", session)
    store = S3ObjectStore(endpoint=endpoint, bucket="my-bucket", access_key_id="", secret_access_key="")
    expected = {"endpoint_url": endpoint} if endpoint else {}
    session.return_value.client.assert_called_once_with("s3", **expected)
    client.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": "a"}]}]
    assert [obj.key for obj in store.iter_objects("prefix/")] == ["a"]
    client.get_paginator.return_value.paginate.assert_called_once_with(Bucket="my-bucket", Prefix="prefix/")
