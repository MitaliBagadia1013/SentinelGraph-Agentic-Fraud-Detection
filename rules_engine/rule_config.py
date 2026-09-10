STOLEN_CARDS_SAMPLE = {"4532123456789012", "5425233430109903"}
FRAUD_DEVICE_IDS_SAMPLE = {"device_fraud_001", "device_fraud_002"}
FRAUD_EMAIL_DOMAINS = {
    "tempmail.com",
    "10minutemail.com",
    "guerrillamail.com",
    "throwaway.email",
    "fakeinbox.com",
}
SANCTIONED_COUNTRIES = {"KP", "IR", "SY", "CU"}
HIGH_RISK_COUNTRIES = {"NG", "GH", "RO", "BG"}
HIGH_RISK_MERCHANT_CATEGORIES = {"5967", "5966", "7995", "6051"}
VELOCITY_LIMITS = {
    "max_transactions_per_minute": 5,
    "max_transactions_per_hour": 20,
    "max_transactions_per_day": 50,
    "max_failed_attempts_per_hour": 3,
    "max_declined_cards_per_hour": 5,
    "max_unique_cards_per_hour": 3,
}
AMOUNT_LIMITS = {
    "new_account_max_amount": 500.0,
    "first_transaction_max_amount": 1000.0,
    "suspicious_round_amounts": [100, 200, 500, 1000, 5000],
    "micro_transaction_threshold": 5.0,
}
LOCATION_RULES = {
    "impossible_travel_speed_kmh": 800,
    "impossible_travel_window_hours": 2,
    "max_countries_per_day": 3,
    "max_distance_from_home_km": 10000,
}
DEVICE_RULES = {
    "allow_new_device": True,
    "max_devices_per_day": 3,
    "require_2fa_for_new_device": True,
    "block_vpn": False,
    "block_tor": True,
}
BEHAVIORAL_RULES = {
    "min_account_age_hours_for_high_value": 24,
    "high_value_threshold": 1000.0,
    "unusual_hour_start": 2,
    "unusual_hour_end": 6,
    "max_merchant_switches_per_hour": 10,
}
INSTANT_APPROVE_RULES = {
    "vip_user_max_amount": 500.0,
    "micro_transaction_auto_approve": True,
    "trusted_merchant_auto_approve": True,
    "same_device_same_location_threshold": 0.95,
}
VIP_USERS = {"user_vip_001", "user_vip_002"}
TRUSTED_MERCHANTS = {"merchant_amazon", "merchant_walmart", "merchant_target"}
RULE_EXECUTION_ORDER = [
    "check_stolen_card",
    "check_sanctioned_country",
    "check_fraud_device",
    "check_tor",
    "check_email_domain",
    "check_amount_exceeds_limit",
    "check_velocity_limits",
    "check_card_testing_pattern",
    "check_vip_user",
    "check_micro_transaction",
    "check_impossible_travel",
    "check_new_account_risk",
    "check_behavioral_anomalies",
]
ENABLED_RULES = {
    "check_stolen_card": True,
    "check_sanctioned_country": True,
    "check_fraud_device": True,
    "check_tor": True,
    "check_amount_exceeds_limit": True,
    "check_velocity_limits": True,
    "check_impossible_travel": True,
    "check_new_account_risk": True,
    "check_behavioral_anomalies": True,
    "check_vip_user": True,
    "check_micro_transaction": True,
    "check_email_domain": True,
    "check_card_testing_pattern": True,
}
LOGGING_CONFIG = {
    "log_all_decisions": True,
    "log_execution_time": True,
    "alert_on_critical_rules": True,
    "alert_threshold_ms": 5.0,
}
