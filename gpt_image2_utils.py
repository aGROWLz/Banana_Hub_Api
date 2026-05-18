BASE_FIELDS = ("model", "prompt", "n", "size")
OFFICIAL_ONLY_FIELDS = ("quality", "response_format", "style", "user")


def is_official_mode(provider_config):
    return provider_config.get("gpt_image2_mode") == "official"


def _clean_body(body):
    return {key: value for key, value in body.items() if value not in (None, "")}


def build_request_payload(
    provider_config,
    *,
    prompt,
    model,
    size,
    n=1,
    quality=None,
    response_format=None,
    style=None,
    user=None,
):
    request_format = provider_config.get("request_format", {}).get("draw", {})
    content_type = request_format.get("content_type", "application/json")
    template = request_format.get("body", {})
    official_mode = is_official_mode(provider_config)

    values = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "quality": quality,
        "response_format": response_format,
        "style": style,
        "user": user,
    }
    allowed_fields = set(BASE_FIELDS)
    if official_mode:
        allowed_fields.update(OFFICIAL_ONLY_FIELDS)

    body = {}
    for field in template:
        if field in allowed_fields:
            body[field] = values.get(field)

    if not body:
        body = {field: values[field] for field in BASE_FIELDS}

    return {
        "content_type": content_type,
        "body": _clean_body(body),
        "official_mode": official_mode,
    }
