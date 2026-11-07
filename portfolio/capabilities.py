from . import data_apps, ml_apps, json_input
from .validation import ValidationError

def enrich(path, data, result):
    """Attach diagnostics without changing existing response fields."""
    tables, notes = ([], [])
    if tables or notes:
        result['capabilities'] = {'tables': tables, 'notes': notes}
    return result

def validated_model(model):
    algorithm = 'standardized binary logistic regression'
    if isinstance(model, dict) and model.get('algorithm', algorithm) != algorithm:
        raise ValidationError('Unsupported model algorithm')
    if not isinstance(model, dict):
        raise ValidationError('Provide a trained model object')
    features = model.get('features')
    if not isinstance(features, list) or not 1 <= len(features) <= 12 or any((not isinstance(f, str) or not f.strip() or len(f) > 1000 for f in features)) or (len(set(features)) != len(features)):
        raise ValidationError('Model requires 1–12 unique feature names')
    vectors = []
    for key in ('weights', 'means', 'scales'):
        values = model.get(key)
        if not isinstance(values, list) or len(values) != len(features):
            raise ValidationError(f'Model {key} must match features')
        vectors.append([ml_apps.number(v, key, 1e-12 if key == 'scales' else -1000000000000.0, 1000000000000.0) for v in values])
    weights, means, scales = vectors
    intercept = ml_apps.number(model.get('intercept'), 'intercept', -1000000000000.0, 1000000000000.0)
    threshold = ml_apps.number(model.get('threshold', 0.5), 'threshold', 0, 1)
    return {'algorithm': algorithm, 'features': features, 'weights': weights, 'means': means, 'scales': scales, 'intercept': intercept, 'threshold': threshold}

def load_model(data):
    raw = data.get('text')
    if not isinstance(raw, str) or not raw.strip() or len(raw.encode('utf-8')) > 200000:
        raise ValidationError('Choose a model JSON file under 200 KB')
    try:
        parsed = json_input.decode(raw)
    except ValueError:
        raise ValidationError('Invalid model JSON: use unique fields and finite numbers') from None
    if isinstance(parsed, dict) and 'model' in parsed:
        parsed = parsed['model']
    return {'model': validated_model(parsed)}

def predict(data):
    model = validated_model(data.get('model'))
    features = model['features']
    weights, means, scales = (model[key] for key in ('weights', 'means', 'scales'))
    intercept = model['intercept']
    threshold = ml_apps.number(data.get('threshold', model['threshold']), 'threshold', 0, 1)
    headers, rows = data_apps.parse(data)
    if any((f not in headers for f in features)):
        raise ValidationError('Prediction CSV must include every model feature')
    predictions = []
    for index, row in enumerate(rows, 2):
        values = [ml_apps.number(row[headers.index(f)], f, -1000000000.0, 1000000000.0) for f in features]
        contributions = [w * (v - mean) / scale for w, v, mean, scale in zip(weights, values, means, scales)]
        score = ml_apps.sigmoid(intercept + sum(contributions))
        predictions.append({'record': index, 'score': score, 'predicted': int(score >= threshold)})
    return {'threshold': threshold, 'rows': len(rows), 'predictions': predictions, 'csv': data_apps.safe_csv(['record', 'score', 'predicted'], [[str(p[k]) for k in ('record', 'score', 'predicted')] for p in predictions])}
