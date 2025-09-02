import numpy as np

from pgmtwin.toolkits.shm.domain import SingleDamageDomain


def get_test_data():
    n_dmg_locs = 3
    n_dmg_lvls = 6

    domain = SingleDamageDomain(
        damage_locations=np.arange(n_dmg_locs + 1),
        damage_levels=np.linspace(0, 1, n_dmg_lvls),
    )

    idxs = np.array([0, 1, 5, 12, len(domain) - 1])
    m_idxs = np.array([[0, 0], [1, 1], [1, 5], [3, 2], [3, 5]])
    values = np.array([[0.0, 0.0], [1.0, 0.2], [1.0, 1.0], [3.0, 0.4], [3.0, 1.0]])

    return domain, idxs, m_idxs, values


# region index
def test_domain_index2values_d0():
    domain, idxs, _, _ = get_test_data()
    ret = domain.index2values(idxs[0])

    assert ret.ndim == 1
    assert np.allclose(ret, idxs[0])


def test_domain_index2values_d1():
    domain, idxs, _, values = get_test_data()
    ret = domain.index2values(idxs)

    assert ret.ndim == 2
    assert np.allclose(ret, values)


def test_domain_index2multi_index_d0():
    domain, idxs, m_idxs, _ = get_test_data()
    ret = domain.index2multi_index(idxs[0])

    assert ret.ndim == 1
    assert np.allclose(ret, m_idxs[0])


def test_domain_index2multi_index_d1():
    domain, idxs, m_idxs, _ = get_test_data()
    ret = domain.index2multi_index(idxs)

    assert ret.ndim == 2
    assert np.allclose(ret, m_idxs)


# endregion


# region values
def test_domain_values2index_d1():
    domain, idxs, _, values = get_test_data()
    ret = domain.values2index(values[0])

    assert ret.ndim == 0
    assert np.allclose(ret, idxs[0])


def test_domain_values2index_d2():
    domain, idxs, _, values = get_test_data()
    ret = domain.values2index(values)

    assert ret.ndim == 1
    assert np.allclose(ret, idxs)


def test_domain_values2multi_index_d1():
    domain, _, m_idxs, values = get_test_data()
    ret = domain.values2multi_index(values[0])

    assert ret.ndim == 1
    assert np.allclose(ret, m_idxs[0])


def test_domain_values2multi_index_d2():
    domain, _, m_idxs, values = get_test_data()
    ret = domain.values2multi_index(values)

    assert ret.ndim == 2
    assert np.allclose(ret, m_idxs)


# endregion


# region multi_index
def test_domain_multi_index2index_d1():
    domain, idxs, m_idxs, _ = get_test_data()
    ret = domain.multi_index2index(m_idxs[0])

    assert ret.ndim == 0
    assert np.allclose(ret, idxs[0])


def test_domain_multi_index2index_d2():
    domain, idxs, m_idxs, _ = get_test_data()
    ret = domain.multi_index2index(m_idxs)

    assert ret.ndim == 1
    assert np.allclose(ret, idxs)


def test_domain_multi_index2values_d1():
    domain, _, m_idxs, values = get_test_data()
    ret = domain.multi_index2values(m_idxs[0])

    assert ret.ndim == 1
    assert np.allclose(ret, values[0])


def test_domain_multi_index2values_d2():
    domain, _, m_idxs, values = get_test_data()
    ret = domain.multi_index2values(m_idxs)

    assert ret.ndim == 2
    assert np.allclose(ret, values)


# endregion
