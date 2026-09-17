"""boto3 source interception, driven through botocore's real event system.

These do not mock the hooks. They build a real boto3 Session, register the real
handlers on it, and emit the two events botocore itself emits with the shapes
botocore itself passes — because the value of this integration is entirely in
whether it agrees with botocore's actual contract, and a test against a mock of
that contract would prove nothing.

The account id is pinned rather than resolved, so nothing here touches STS or
the network.
"""
import pytest

boto3 = pytest.importorskip("boto3")

from ai_telemetry.context import current_user_context
from ai_telemetry.integrations import boto3_integration as bi
from ai_telemetry.integrations.boto3_integration import (
    patch_boto3, drain_sources, set_account_id, _derive,
)

ACCOUNT = "123456789012"
REGION = "us-east-1"


@pytest.fixture(autouse=True)
def _clean():
    set_account_id(ACCOUNT)
    current_user_context.set({})
    yield
    bi._patched_sessions.clear()
    bi._account_id = None
    bi._account_resolved = False


_CLIENTS = {}


def _client(sess, service):
    """A real boto3 client, cached. Creation is local — no network, no creds used."""
    if service not in _CLIENTS:
        _CLIENTS[service] = sess.client(
            service, region_name=REGION,
            aws_access_key_id="x", aws_secret_access_key="y")
    return _CLIENTS[service]


class _Response:
    def __init__(self, status_code=200):
        self.status_code = status_code


def _session():
    sess = boto3.Session(region_name=REGION,
                         aws_access_key_id="x", aws_secret_access_key="y")
    assert patch_boto3(session=sess) is True
    return sess


def _call(sess, service, operation, params, status=200, parsed=None):
    """Emit the two events botocore emits, with botocore's real OperationModel.

    A stand-in model is not good enough here: botocore registers its own
    handlers on these same events and reads fields off the model that only a
    real one has. Emitting against the genuine article is the only way this
    proves the integration agrees with botocore's actual contract.
    """
    client = _client(sess, service)
    model = client.meta.service_model.operation_model(operation)
    service_id = str(model.service_model.service_id).lower()
    context = {}
    client.meta.events.emit(
        f"before-parameter-build.{service_id}.{operation}",
        params=params, model=model, context=context)
    client.meta.events.emit(
        f"after-call.{service_id}.{operation}",
        http_response=_Response(status), parsed=parsed if parsed is not None else {},
        model=model, context=context)
    return context


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

class TestResourceMapping:

    @pytest.mark.parametrize("service,op,params,expected", [
        ("s3", "GetObject", {"Bucket": "customer-data-bucket", "Key": "reports/q3.pdf"},
         "arn:aws:s3:::customer-data-bucket/reports/q3.pdf"),
        ("s3", "ListObjectsV2", {"Bucket": "customer-data-bucket"},
         "arn:aws:s3:::customer-data-bucket"),
        ("s3", "ListObjectsV2", {"Bucket": "b", "Prefix": "2026/"},
         "arn:aws:s3:::b/2026/"),
        ("dynamodb", "GetItem", {"TableName": "CustomerAccounts"},
         f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/CustomerAccounts"),
        ("dynamodb", "Query", {"TableName": "CustomerAccounts", "IndexName": "by-email"},
         f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/CustomerAccounts/index/by-email"),
        ("ssm", "GetParameter", {"Name": "/prod/db/password"},
         f"arn:aws:ssm:{REGION}:{ACCOUNT}:parameter/prod/db/password"),
        ("ssm", "GetParameter", {"Name": "flat-name"},
         f"arn:aws:ssm:{REGION}:{ACCOUNT}:parameter/flat-name"),
        ("bedrock agent runtime", "Retrieve", {"knowledgeBaseId": "KB123"},
         f"arn:aws:bedrock:{REGION}:{ACCOUNT}:knowledge-base/KB123"),
        ("kendra", "Query", {"IndexId": "idx-1"},
         f"arn:aws:kendra:{REGION}:{ACCOUNT}:index/idx-1"),
        ("rds data", "ExecuteStatement",
         {"resourceArn": f"arn:aws:rds:{REGION}:{ACCOUNT}:cluster:fin"},
         f"arn:aws:rds:{REGION}:{ACCOUNT}:cluster:fin"),
        ("athena", "StartQueryExecution",
         {"QueryExecutionContext": {"Database": "analytics"}},
         f"arn:aws:athena:{REGION}:{ACCOUNT}:database/analytics"),
    ])
    def test_operation_maps_to_arn(self, service, op, params, expected):
        assert _derive(service, op, params, REGION, ACCOUNT) == [expected]

    def test_s3_arns_carry_no_region_or_account(self):
        """S3 ARNs are genuinely global. Adding either would produce a string
        that never matches what an IAM policy granted."""
        uri = _derive("s3", "GetObject", {"Bucket": "b", "Key": "k"}, REGION, ACCOUNT)[0]
        assert uri == "arn:aws:s3:::b/k"
        assert REGION not in uri and ACCOUNT not in uri

    def test_secret_arn_is_passed_through_untouched(self):
        """AWS appends a random suffix to every secret ARN. Rebuilding one from
        the name would produce a string IAM never granted."""
        full = f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:db-creds-AbCdEf"
        assert _derive("secrets manager", "GetSecretValue",
                       {"SecretId": full}, REGION, ACCOUNT) == [full]

    def test_bare_secret_name_is_expanded(self):
        got = _derive("secrets manager", "GetSecretValue",
                      {"SecretId": "db-creds"}, REGION, ACCOUNT)[0]
        assert got == f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:db-creds"

    def test_batch_get_item_reports_every_table(self):
        """Reporting only the first would silently drop the rest from the chain."""
        got = _derive("dynamodb", "BatchGetItem",
                      {"RequestItems": {"Customers": {}, "Orders": {}}}, REGION, ACCOUNT)
        assert sorted(got) == [
            f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/Customers",
            f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/Orders",
        ]

    def test_writes_are_not_retrievals(self):
        """PutObject contributes no bytes to a context window. Recording it
        would put a source on the chain that the agent never read."""
        assert _derive("s3", "PutObject", {"Bucket": "b", "Key": "k"}, REGION, ACCOUNT) == []
        assert _derive("dynamodb", "PutItem", {"TableName": "T"}, REGION, ACCOUNT) == []
        assert _derive("dynamodb", "DeleteItem", {"TableName": "T"}, REGION, ACCOUNT) == []

    def test_unmapped_service_is_ignored(self):
        assert _derive("cloudwatch", "GetMetricData", {}, REGION, ACCOUNT) == []


# ---------------------------------------------------------------------------
# The two-event handshake
# ---------------------------------------------------------------------------

class TestEventHandshake:

    def test_successful_call_records_the_source(self):
        sess = _session()
        _call(sess, "dynamodb", "GetItem", {"TableName": "CustomerAccounts"})
        assert drain_sources() == [
            f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/CustomerAccounts"]

    def test_access_denied_records_nothing(self):
        """The one case where the agent demonstrably did not receive the data.
        Recording it would attribute content to a source that returned none."""
        sess = _session()
        _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"},
              status=403, parsed={"Error": {"Code": "AccessDenied"}})
        assert drain_sources() == []

    def test_error_in_parsed_body_records_nothing(self):
        sess = _session()
        _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"},
              status=200, parsed={"Error": {"Code": "NoSuchKey"}})
        assert drain_sources() == []

    def test_context_is_cleared_after_promotion(self):
        """The stash must not leak into a later request reusing the dict."""
        sess = _session()
        ctx = _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"})
        assert bi._CTX_KEY not in ctx

    def test_several_reads_in_one_tool_call_all_land(self):
        """A tool that looks up an account then fetches a document has read two
        sources, and both belong on that tool call's provenance chain."""
        sess = _session()
        _call(sess, "dynamodb", "GetItem", {"TableName": "CustomerAccounts"})
        _call(sess, "s3", "GetObject", {"Bucket": "docs", "Key": "policy.pdf"})
        assert drain_sources() == [
            f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/CustomerAccounts",
            "arn:aws:s3:::docs/policy.pdf",
        ]

    def test_repeated_reads_of_one_source_record_once(self):
        sess = _session()
        for _ in range(3):
            _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"})
        assert drain_sources() == ["arn:aws:s3:::b/k"]

    def test_draining_twice_yields_nothing_the_second_time(self):
        sess = _session()
        _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"})
        assert drain_sources()
        assert drain_sources() == []

    def test_sts_is_never_instrumented(self):
        """The account resolver calls STS; instrumenting it would recurse."""
        sess = _session()
        _call(sess, "sts", "GetCallerIdentity", {})
        assert drain_sources() == []

    def test_registration_is_idempotent(self):
        sess = boto3.Session(region_name=REGION,
                             aws_access_key_id="x", aws_secret_access_key="y")
        assert patch_boto3(session=sess) is True
        assert patch_boto3(session=sess) is True
        _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"})
        assert drain_sources() == ["arn:aws:s3:::b/k"]


# ---------------------------------------------------------------------------
# The hooks must never break the host's AWS calls
# ---------------------------------------------------------------------------

class TestHooksAreHarmless:

    def test_malformed_params_do_not_raise(self):
        sess = _session()
        _call(sess, "dynamodb", "GetItem", {"TableName": None})
        _call(sess, "s3", "GetObject", {})
        _call(sess, "dynamodb", "BatchGetItem", {"RequestItems": None})
        assert drain_sources() == []

    def test_the_hooks_tolerate_a_missing_context(self):
        """Called directly rather than through the event system, deliberately.

        botocore always passes a real context dict — its own handlers assume one
        and raise without it — so emitting context=None through the event system
        would test botocore's tolerance, not ours. The guard still earns its
        place: these hooks are registered on a wildcard and will be handed
        whatever any future botocore version, or another library's re-emit,
        chooses to pass.
        """
        sess = _session()
        before = bi._make_before_hook(sess, debug=False)
        after = bi._make_after_hook(debug=False)

        before(params={"Bucket": "b"}, model=None, context=None)
        after(http_response=_Response(), parsed={}, context=None)
        before()
        after()
        assert drain_sources() == []

    def test_a_broken_derivation_does_not_break_the_call(self, monkeypatch):
        """A hook that throws would break the host application's AWS call —
        an unacceptable price for telemetry."""
        sess = _session()
        monkeypatch.setattr(bi, "_derive",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        _call(sess, "s3", "GetObject", {"Bucket": "b", "Key": "k"})   # must not raise
        assert drain_sources() == []


# ---------------------------------------------------------------------------
# End to end: interception feeds the scope boundary
# ---------------------------------------------------------------------------

def test_intercepted_read_of_an_unpermitted_table_fires_critical():
    """The whole point: no annotation anywhere, and the boundary still fires.

    Only boto3 is told what to do. No source_uri on a decorator, no
    source_context block, no data_sources string.
    """
    import os, tempfile
    import ai_telemetry
    from ai_telemetry.client import TelemetryEvent
    from ai_telemetry.scope_checker import SC_OUT_OF_SCOPE_SOURCE
    from ai_telemetry import scope_runtime

    scope_runtime._seen_sessions.clear()
    scope_runtime._checker._session_hashes.clear()

    client = ai_telemetry.init(
        app_id="document-qa-agent", api_key="sk-test",
        endpoint="http://127.0.0.1:1", enable_local_log=True,
        log_path=os.path.join(tempfile.mkdtemp(), "events.csv"),
        permitted_sources=["arn:aws:s3:::internal-docs"],
        permitted_classifications=["INTERNAL", "PUBLIC"],
    )
    client._send_http = lambda events: None

    sess = _session()
    _call(sess, "dynamodb", "GetItem", {"TableName": "CustomerAccounts"})

    ev = TelemetryEvent(event_type="tool_call", tool_name="lookup_account",
                        tool_output="{'name': 'Alice'}")
    client.capture(ev)
    client._send([ev])

    assert ev.source_uri == f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/CustomerAccounts"
    assert ev.source_type == "database"
    assert ev.source_in_manifest is False
    assert ev.scope_signals == SC_OUT_OF_SCOPE_SOURCE
    assert ev.scope_severity == "CRITICAL"


def test_intercepted_read_of_a_permitted_bucket_is_clean():
    """The false-positive case, through the real matcher: IAM grants the bucket,
    boto3 reports the object key."""
    import os, tempfile
    import ai_telemetry
    from ai_telemetry.client import TelemetryEvent
    from ai_telemetry import scope_runtime

    scope_runtime._seen_sessions.clear()
    scope_runtime._checker._session_hashes.clear()

    client = ai_telemetry.init(
        app_id="cs-agent", api_key="sk-test", endpoint="http://127.0.0.1:1",
        enable_local_log=True,
        log_path=os.path.join(tempfile.mkdtemp(), "events.csv"),
        asset_record={
            "arn": "arn:aws:lambda:us-east-1:123456789012:function:cs-agent",
            "account_id": ACCOUNT,
            "data_sources_permitted": ["arn:aws:s3:::customer-data-bucket"],
            "pii_classification": "HIGH",
        },
    )
    client._send_http = lambda events: None

    sess = _session()
    _call(sess, "s3", "GetObject",
          {"Bucket": "customer-data-bucket", "Key": "reports/q3.pdf"})

    ev = TelemetryEvent(event_type="tool_call", tool_name="read_report",
                        tool_output="quarterly figures")
    client.capture(ev)
    client._send([ev])

    assert ev.source_uri == "arn:aws:s3:::customer-data-bucket/reports/q3.pdf"
    assert ev.source_in_manifest is True
    assert ev.scope_signals is None
