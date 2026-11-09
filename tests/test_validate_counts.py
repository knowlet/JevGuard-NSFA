import pytest
from test_validate_report_shapes import valid_report, validate_report, write_report


@pytest.mark.parametrize('field', ['tp', 'tn', 'fp', 'fn'])
@pytest.mark.parametrize('value', [True, '1', 1.5, -1, float('inf')])
def test_invalid_confusion_count(tmp_path, field, value):
    data = valid_report()
    data['quality']['binary'][field] = value
    result = validate_report(write_report(tmp_path, data))
    assert not result['ok']
    assert any('confusion matrix' in e for e in result['errors'])

@pytest.mark.parametrize('value', [0, 1, 10**400])
def test_valid_integer_counts(tmp_path, value):
    data = valid_report()
    for field in ['tp', 'tn', 'fp', 'fn']:
        data['quality']['binary'][field] = value
    data['samples'].update(attempted=4*value, successful=4*value)
    if value == 0:
        for field in ['accuracy', 'precision', 'recall', 'f1']:
            data['quality']['binary'][field] = 0
    assert validate_report(write_report(tmp_path, data))['ok']
