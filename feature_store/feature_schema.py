from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: type
    group: str
    default: Any
    description: str = ""


FEATURES: List[Feature] = [
    Feature(
        "avg_transaction_amount",
        float,
        "velocity",
        0.0,
        "User's overall average transaction amount",
    ),
    Feature(
        "transaction_count", int, "velocity", 0, "Total lifetime transaction count"
    ),
    Feature(
        "transactions_last_hour", int, "velocity", 0, "Transactions in last 1 hour"
    ),
    Feature(
        "transactions_last_day", int, "velocity", 0, "Transactions in last 24 hours"
    ),
    Feature(
        "transactions_last_week", int, "velocity", 0, "Transactions in last 7 days"
    ),
    Feature(
        "amount_last_hour", float, "velocity", 0.0, "Total amount spent in last 1 hour"
    ),
    Feature(
        "amount_last_day", float, "velocity", 0.0, "Total amount spent in last 24 hours"
    ),
    Feature(
        "amount_last_week", float, "velocity", 0.0, "Total amount spent in last 7 days"
    ),
    Feature(
        "avg_amount_last_7_days",
        float,
        "velocity",
        0.0,
        "Average transaction amount over last 7 days",
    ),
    Feature(
        "max_amount_last_30_days",
        float,
        "velocity",
        0.0,
        "Maximum single transaction in last 30 days",
    ),
    Feature(
        "std_amount_last_30_days",
        float,
        "velocity",
        0.0,
        "Std deviation of amounts in last 30 days",
    ),
    Feature(
        "declined_transaction_count",
        int,
        "velocity",
        0,
        "Number of declined transactions",
    ),
    Feature("chargeback_count", int, "velocity", 0, "Number of chargebacks raised"),
    Feature("refund_count", int, "velocity", 0, "Number of refunds requested"),
    Feature(
        "transaction_diversity_score",
        float,
        "velocity",
        0.0,
        "Diversity of merchants/categories transacted at",
    ),
    Feature(
        "avg_days_between_transactions",
        float,
        "velocity",
        30.0,
        "Average gap in days between transactions",
    ),
    Feature(
        "weekend_transaction_ratio",
        float,
        "velocity",
        0.0,
        "Fraction of transactions on weekends",
    ),
    Feature(
        "night_transaction_ratio",
        float,
        "velocity",
        0.0,
        "Fraction of transactions between 10pm-6am",
    ),
    Feature(
        "cross_border_transaction_ratio",
        float,
        "velocity",
        0.0,
        "Fraction of transactions in foreign country",
    ),
    Feature(
        "large_transaction_count",
        int,
        "velocity",
        0,
        "Count of transactions above large-amount threshold",
    ),
    Feature(
        "transaction_amount_z_score",
        float,
        "velocity",
        0.0,
        "Z-score of current amount vs user history",
    ),
    Feature("device_count", int, "device", 1, "Number of unique devices used"),
    Feature(
        "browser_type",
        str,
        "device",
        "unknown",
        "Browser family (Chrome, Firefox, etc.)",
    ),
    Feature(
        "os_type", str, "device", "unknown", "Operating system (Windows, iOS, etc.)"
    ),
    Feature("is_mobile", int, "device", 0, "1 if transaction from mobile device"),
    Feature("is_vpn", int, "device", 0, "1 if VPN detected"),
    Feature("is_proxy", int, "device", 0, "1 if proxy detected"),
    Feature("is_tor", int, "device", 0, "1 if Tor exit node detected"),
    Feature("ip_reputation_score", float, "device", 0.5, "IP reputation 0=bad, 1=good"),
    Feature(
        "device_age_days", int, "device", 0, "Days since this device was first seen"
    ),
    Feature("screen_resolution_width", int, "device", 1920, "Screen width in pixels"),
    Feature("screen_resolution_height", int, "device", 1080, "Screen height in pixels"),
    Feature("browser_language", str, "device", "en", "Browser language code"),
    Feature("plugins_count", int, "device", 0, "Number of browser plugins"),
    Feature("cookies_enabled", int, "device", 1, "1 if cookies are enabled"),
    Feature("javascript_enabled", int, "device", 1, "1 if JavaScript is enabled"),
    Feature(
        "session_duration_seconds",
        int,
        "device",
        0,
        "Length of current session in seconds",
    ),
    Feature(
        "device_fingerprint_changes",
        int,
        "device",
        0,
        "Number of fingerprint changes in last 30 days",
    ),
    Feature(
        "ip_changes_last_week",
        int,
        "device",
        0,
        "Number of IP address changes in last 7 days",
    ),
    Feature(
        "unique_ips_last_month",
        int,
        "device",
        1,
        "Number of unique IPs used in last 30 days",
    ),
    Feature(
        "device_os_mismatch", int, "device", 0, "1 if device OS contradicts user agent"
    ),
    Feature(
        "browser_version_outdated",
        int,
        "device",
        0,
        "1 if browser version is significantly outdated",
    ),
    Feature(
        "timezone_mismatch",
        int,
        "device",
        0,
        "1 if device timezone contradicts billing location",
    ),
    Feature(
        "user_agent_anomaly_score",
        float,
        "device",
        0.0,
        "Anomaly score of user agent string",
    ),
    Feature(
        "device_reputation_score",
        float,
        "device",
        0.5,
        "Device reputation 0=bad, 1=good",
    ),
    Feature(
        "incognito_mode_usage_ratio",
        float,
        "device",
        0.0,
        "Fraction of sessions in incognito/private mode",
    ),
    Feature(
        "screen_orientation_changes",
        int,
        "device",
        0,
        "Number of screen orientation changes in session",
    ),
    Feature("touch_capability", int, "device", 0, "1 if device has touch screen"),
    Feature(
        "webrtc_enabled", int, "device", 0, "1 if WebRTC is enabled (IP leak risk)"
    ),
    Feature(
        "canvas_fingerprint_changes",
        int,
        "device",
        0,
        "Number of canvas fingerprint changes",
    ),
    Feature(
        "time_on_site_seconds",
        int,
        "behavioral",
        0,
        "Total time on site before purchase",
    ),
    Feature("pages_viewed", int, "behavioral", 0, "Number of pages viewed in session"),
    Feature("products_viewed", float, "behavioral", 0.0, "Number of products viewed"),
    Feature("cart_additions", int, "behavioral", 0, "Number of items added to cart"),
    Feature("cart_removals", int, "behavioral", 0, "Number of items removed from cart"),
    Feature(
        "form_fill_time_seconds", int, "behavioral", 0, "Time to fill checkout form"
    ),
    Feature(
        "paste_events_count",
        int,
        "behavioral",
        0,
        "Number of paste events (card details pasted)",
    ),
    Feature(
        "autocomplete_usage_count",
        int,
        "behavioral",
        0,
        "Number of autocomplete suggestions used",
    ),
    Feature(
        "typing_speed_wpm",
        float,
        "behavioral",
        0.0,
        "Average typing speed in words per minute",
    ),
    Feature(
        "mouse_movement_speed", float, "behavioral", 0.0, "Average mouse movement speed"
    ),
    Feature(
        "click_hesitation_avg_ms",
        float,
        "behavioral",
        0.0,
        "Average ms between hover and click",
    ),
    Feature(
        "scroll_events_count",
        int,
        "behavioral",
        0,
        "Number of scroll events in session",
    ),
    Feature("scroll_depth_avg", float, "behavioral", 0.0, "Average scroll depth (0-1)"),
    Feature("back_button_clicks", int, "behavioral", 0, "Number of back button clicks"),
    Feature(
        "failed_login_attempts",
        int,
        "behavioral",
        0,
        "Failed login attempts before success",
    ),
    Feature("password_reset_count", int, "behavioral", 0, "Number of password resets"),
    Feature(
        "session_replay_anomaly_score",
        float,
        "behavioral",
        0.0,
        "ML score from session replay analysis",
    ),
    Feature(
        "bot_detection_score",
        float,
        "behavioral",
        0.0,
        "Bot probability 0=human, 1=bot",
    ),
    Feature(
        "keyboard_rhythm_consistency",
        float,
        "behavioral",
        0.5,
        "Consistency of typing rhythm (biometric)",
    ),
    Feature(
        "mouse_trajectory_smoothness",
        float,
        "behavioral",
        0.5,
        "Smoothness of mouse movements",
    ),
    Feature(
        "form_field_order_anomaly",
        int,
        "behavioral",
        0,
        "1 if form filled in unexpected order",
    ),
    Feature(
        "copy_events_count", int, "behavioral", 0, "Number of copy events in session"
    ),
    Feature("right_click_count", int, "behavioral", 0, "Number of right-click events"),
    Feature("tab_switches", int, "behavioral", 0, "Number of tab/window switches"),
    Feature("idle_time_seconds", int, "behavioral", 0, "Total idle time in session"),
    Feature(
        "shared_device_count",
        int,
        "network",
        0,
        "Number of other accounts sharing this device",
    ),
    Feature("shared_ip_count", int, "network", 0, "Number of accounts sharing this IP"),
    Feature(
        "shared_billing_address_count",
        int,
        "network",
        0,
        "Accounts with same billing address",
    ),
    Feature(
        "shared_shipping_address_count",
        int,
        "network",
        0,
        "Accounts with same shipping address",
    ),
    Feature(
        "email_domain", str, "network", "unknown", "Email domain (gmail.com, etc.)"
    ),
    Feature("email_age_days", int, "network", 0, "Age of email address in days"),
    Feature(
        "email_domain_reputation", float, "network", 0.5, "Reputation of email domain"
    ),
    Feature("phone_verified", int, "network", 0, "1 if phone number is verified"),
    Feature(
        "phone_carrier_type",
        str,
        "network",
        "unknown",
        "Phone carrier type (mobile, voip, landline)",
    ),
    Feature(
        "social_media_linked", int, "network", 0, "1 if social media account linked"
    ),
    Feature(
        "referral_source",
        str,
        "network",
        "direct",
        "How user arrived (organic, ad, referral)",
    ),
    Feature(
        "marketing_channel",
        str,
        "network",
        "unknown",
        "Marketing channel that acquired user",
    ),
    Feature(
        "connected_accounts_count",
        int,
        "network",
        0,
        "Number of accounts connected to this user",
    ),
    Feature(
        "network_centrality_score",
        float,
        "network",
        0.0,
        "Graph centrality — high = hub in fraud network",
    ),
    Feature(
        "shared_payment_method_count",
        int,
        "network",
        0,
        "Other accounts using same payment method",
    ),
    Feature(
        "email_pattern_similarity_score",
        float,
        "network",
        0.0,
        "Similarity to known synthetic email patterns",
    ),
    Feature(
        "phone_shared_count",
        int,
        "network",
        0,
        "Number of accounts sharing this phone number",
    ),
    Feature(
        "address_history_count", int, "network", 0, "Number of distinct addresses used"
    ),
    Feature(
        "account_linking_anomaly_score",
        float,
        "network",
        0.0,
        "Anomaly score from account linking patterns",
    ),
    Feature(
        "graph_community_fraud_rate",
        float,
        "network",
        0.0,
        "Fraud rate of user's graph community",
    ),
    Feature("location_lat", float, "geo", 0.0, "Transaction latitude"),
    Feature("location_lon", float, "geo", 0.0, "Transaction longitude"),
    Feature("ip_country", str, "geo", "unknown", "Country derived from IP address"),
    Feature("billing_country", str, "geo", "unknown", "Country from billing address"),
    Feature("shipping_country", str, "geo", "unknown", "Country from shipping address"),
    Feature(
        "is_high_risk_country", int, "geo", 0, "1 if transaction country is high-risk"
    ),
    Feature("country_risk_score", float, "geo", 0.0, "Country-level fraud risk score"),
    Feature(
        "distance_from_last_transaction_km",
        float,
        "geo",
        0.0,
        "KM distance from previous transaction location",
    ),
    Feature(
        "billing_shipping_distance_km",
        float,
        "geo",
        0.0,
        "KM distance between billing and shipping address",
    ),
    Feature(
        "impossible_travel_flag",
        int,
        "geo",
        0,
        "1 if travel speed between transactions is impossible",
    ),
    Feature(
        "location_consistency_score",
        float,
        "geo",
        1.0,
        "How consistent location is with user history",
    ),
    Feature("timezone", str, "geo", "UTC", "User's timezone"),
    Feature(
        "ip_billing_country_match",
        int,
        "geo",
        1,
        "1 if IP country matches billing country",
    ),
    Feature("is_international", int, "geo", 0, "1 if transaction is international"),
    Feature(
        "address_verification_status",
        str,
        "geo",
        "unknown",
        "AVS result (match, partial, no_match)",
    ),
    Feature("zip_code_match", int, "geo", 1, "1 if zip code matches billing address"),
    Feature(
        "urban_vs_rural",
        str,
        "geo",
        "unknown",
        "Urban or rural classification of location",
    ),
    Feature(
        "location_change_frequency",
        float,
        "geo",
        0.0,
        "How often user transacts from new locations",
    ),
    Feature(
        "avg_distance_per_transaction",
        float,
        "geo",
        0.0,
        "Average KM distance between consecutive transactions",
    ),
    Feature(
        "max_distance_last_month_km",
        float,
        "geo",
        0.0,
        "Max distance travelled in last 30 days",
    ),
    Feature("account_age_days", int, "identity", 0, "Days since account was created"),
    Feature("email_verified", int, "identity", 0, "1 if email address is verified"),
    Feature(
        "identity_verified",
        int,
        "identity",
        0,
        "1 if full identity verification passed (KYC)",
    ),
    Feature(
        "customer_lifetime_value",
        float,
        "identity",
        0.0,
        "Total spend over account lifetime",
    ),
    Feature(
        "loyalty_program_member", int, "identity", 0, "1 if enrolled in loyalty program"
    ),
    Feature("return_rate", float, "identity", 0.0, "Fraction of orders returned"),
    Feature(
        "customer_service_contacts",
        int,
        "identity",
        0,
        "Number of customer service contacts",
    ),
    Feature("complaints_count", int, "identity", 0, "Number of complaints filed"),
    Feature(
        "previous_fraud_flags",
        int,
        "identity",
        0,
        "Number of previous fraud flags on account",
    ),
    Feature(
        "successful_order_rate",
        float,
        "identity",
        1.0,
        "Fraction of orders successfully completed",
    ),
    Feature(
        "profile_completeness_score",
        float,
        "identity",
        0.0,
        "How complete the user profile is (0-1)",
    ),
    Feature(
        "account_updates_count", int, "identity", 0, "Number of account detail changes"
    ),
    Feature(
        "suspicious_activity_count",
        int,
        "identity",
        0,
        "Number of suspicious activity flags",
    ),
    Feature(
        "positive_feedback_count",
        int,
        "identity",
        0,
        "Number of positive reviews/feedback",
    ),
    Feature(
        "negative_feedback_count",
        int,
        "identity",
        0,
        "Number of negative reviews/feedback",
    ),
    Feature(
        "card_type", str, "payment", "unknown", "Card type (credit, debit, prepaid)"
    ),
    Feature(
        "card_brand",
        str,
        "payment",
        "unknown",
        "Card brand (Visa, Mastercard, Amex, etc.)",
    ),
    Feature("is_prepaid_card", int, "payment", 0, "1 if card is prepaid"),
    Feature("card_country", str, "payment", "unknown", "Country of card issuer"),
    Feature(
        "card_billing_country_match",
        int,
        "payment",
        1,
        "1 if card country matches billing country",
    ),
    Feature("cvv_match", int, "payment", 1, "1 if CVV matches"),
    Feature("avs_match", int, "payment", 1, "1 if AVS check passed"),
    Feature(
        "3d_secure_status", str, "payment", "unknown", "3D Secure authentication result"
    ),
    Feature(
        "is_saved_card", int, "payment", 0, "1 if card was previously saved to account"
    ),
    Feature(
        "card_age_days", int, "payment", 0, "Days since card was first used on platform"
    ),
    Feature(
        "failed_payment_count", int, "payment", 0, "Number of failed payment attempts"
    ),
    Feature(
        "card_shared_accounts_count",
        int,
        "payment",
        0,
        "Number of accounts using this card",
    ),
    Feature(
        "card_velocity_24h",
        int,
        "payment",
        0,
        "Number of transactions on this card in last 24h",
    ),
    Feature(
        "card_bin_risk_score",
        float,
        "payment",
        0.0,
        "Risk score for card BIN (first 6 digits)",
    ),
    Feature(
        "card_issuer_country", str, "payment", "unknown", "Country of card issuing bank"
    ),
    Feature(
        "payment_method_changes", int, "payment", 0, "Number of payment method changes"
    ),
    Feature(
        "digital_wallet_used",
        int,
        "payment",
        0,
        "1 if payment via digital wallet (Apple/Google Pay)",
    ),
    Feature("tokenized_payment", int, "payment", 0, "1 if payment is tokenized"),
    Feature(
        "card_expiry_months_remaining", int, "payment", 12, "Months until card expires"
    ),
    Feature(
        "card_issuer_reputation",
        float,
        "payment",
        0.5,
        "Reputation score of card issuing bank",
    ),
    Feature(
        "merchant_category", str, "merchant", "unknown", "Merchant category code (MCC)"
    ),
    Feature(
        "is_high_risk_merchant_category",
        int,
        "merchant",
        0,
        "1 if MCC is in high-risk category list",
    ),
    Feature("merchant_risk_score", float, "merchant", 0.0, "Merchant-level risk score"),
    Feature(
        "is_digital_goods", int, "merchant", 0, "1 if transaction is for digital goods"
    ),
    Feature(
        "is_international_merchant",
        int,
        "merchant",
        0,
        "1 if merchant is in foreign country",
    ),
    Feature(
        "merchant_account_age_days",
        int,
        "merchant",
        0,
        "Days since merchant joined platform",
    ),
    Feature(
        "merchant_chargeback_rate",
        float,
        "merchant",
        0.0,
        "Merchant's historical chargeback rate",
    ),
    Feature(
        "merchant_fraud_rate",
        float,
        "merchant",
        0.0,
        "Merchant's historical fraud rate",
    ),
    Feature(
        "product_price_vs_market_ratio",
        float,
        "merchant",
        1.0,
        "Price vs market average (>2 = suspicious)",
    ),
    Feature(
        "bulk_purchase_flag", int, "merchant", 0, "1 if buying unusually large quantity"
    ),
    Feature(
        "rush_delivery_flag", int, "merchant", 0, "1 if rush/express delivery selected"
    ),
    Feature("gift_card_purchase", int, "merchant", 0, "1 if purchasing gift cards"),
    Feature(
        "merchant_fraud_history",
        int,
        "merchant",
        0,
        "1 if merchant has prior fraud history",
    ),
    Feature(
        "merchant_reputation_score",
        float,
        "merchant",
        0.5,
        "Overall merchant reputation score",
    ),
    Feature("hour_of_day", int, "temporal", 12, "Hour of transaction (0-23)"),
    Feature("day_of_week", int, "temporal", 1, "Day of week (0=Monday, 6=Sunday)"),
    Feature("is_weekend", int, "temporal", 0, "1 if transaction on weekend"),
    Feature(
        "is_business_hours",
        int,
        "temporal",
        1,
        "1 if transaction during business hours",
    ),
    Feature("is_night_time", int, "temporal", 0, "1 if transaction between 10pm-6am"),
    Feature(
        "days_since_last_transaction",
        float,
        "temporal",
        30.0,
        "Days since user's last transaction",
    ),
    Feature(
        "hours_since_last_login",
        float,
        "temporal",
        24.0,
        "Hours since user last logged in",
    ),
    Feature("is_holiday", int, "temporal", 0, "1 if transaction on public holiday"),
    Feature(
        "time_to_checkout_seconds",
        int,
        "temporal",
        0,
        "Seconds from session start to checkout",
    ),
    Feature(
        "session_to_purchase_ratio",
        float,
        "temporal",
        0.0,
        "Ratio of session duration to time-to-purchase",
    ),
]
LABEL_COLUMNS = ["fraud_score", "is_fraud", "fraud_scenario"]
META_COLUMNS = ["user_id", "timestamp", "amount", "location", "device_id"]
FEATURE_MAP: Dict[str, Feature] = {f.name: f for f in FEATURES}
FEATURE_GROUPS: Dict[str, List[Feature]] = {}
for _f in FEATURES:
    FEATURE_GROUPS.setdefault(_f.group, []).append(_f)
FEATURE_NAMES: List[str] = [f.name for f in FEATURES]
FEATURE_DEFAULTS: Dict[str, Any] = {f.name: f.default for f in FEATURES}
NUMERIC_FEATURES: List[str] = [f.name for f in FEATURES if f.dtype in (float, int)]
CATEGORICAL_FEATURES: List[str] = [f.name for f in FEATURES if f.dtype == str]
REDIS_FEATURE_NAMES: List[str] = [
    "transactions_last_hour",
    "transactions_last_day",
    "transactions_last_week",
    "amount_last_hour",
    "amount_last_day",
    "amount_last_week",
    "days_since_last_transaction",
    "transaction_amount_z_score",
]
REDIS_FEATURE_DEFAULTS: Dict[str, Any] = {
    k: FEATURE_DEFAULTS[k] for k in REDIS_FEATURE_NAMES if k in FEATURE_DEFAULTS
}


def get_feature(name: str) -> Optional[Feature]:
    return FEATURE_MAP.get(name)


def get_group_features(group: str) -> List[Feature]:
    return FEATURE_GROUPS.get(group, [])


def apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {f.name: raw.get(f.name, f.default) for f in FEATURES}


def validate_feature_dict(features: Dict[str, Any]) -> List[str]:
    warnings = []
    for f in FEATURES:
        if f.name not in features:
            warnings.append(f"Missing feature: {f.name}")
        elif features[f.name] is not None:
            val = features[f.name]
            if f.dtype == float and (not isinstance(val, (float, int))):
                warnings.append(
                    f"Wrong type for {f.name}: expected float, got {type(val).__name__}"
                )
            elif f.dtype == int and (not isinstance(val, (int, float))):
                warnings.append(
                    f"Wrong type for {f.name}: expected int, got {type(val).__name__}"
                )
    return warnings


def summary() -> str:
    lines = [f"Total features: {len(FEATURES)}"]
    for group, feats in sorted(FEATURE_GROUPS.items()):
        lines.append(f"{group:12s}: {len(feats)} features")
    lines.append(f"{'label':12s}: {len(LABEL_COLUMNS)} columns (excluded from model)")
    lines.append(f"{'meta':12s}: {len(META_COLUMNS)} columns (identifiers)")
    return "\n".join(lines)
