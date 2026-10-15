from magnum.magnum import (
    load_model_file,
    run_model
)
from scipy import stats


def test_client_visits():
    model = load_model_file("models/coffee/queue.mgn")
    open_hour = 8
    close_hour = 21
    average_count_clients_per_1h = 10
    result = run_model(model, controls={
        "average_wait": 60,
        "std_wait": 5,
        "average_count_clients_per_1h": average_count_clients_per_1h,
        "open_hour": open_hour,
        "close_hour": close_hour
    }, horizon=300, sampling='on_change')
    times = (result['variables']['last_client_id']['time'])[1:]
    grouped = {}
    for t in times:
        grouped.setdefault(int(t), []).append(t)
    diffs = []
    for lt in grouped.values():
        diffs.extend([r - l for l, r in zip(lt, lt[1:])])
    _, p_value = stats.ks_2samp(diffs, stats.expon.rvs(size=1000, scale=1/24/average_count_clients_per_1h))
    print(p_value)
    # проверим, что приход клиентов соответствует экспоненциальному распределению
    assert p_value > 0.001
    assert min([t - int(t) for t in times]) > open_hour / 24.0
    assert max([t - int(t) for t in times]) < close_hour / 24.0


def tests():
    test_client_visits()


tests()