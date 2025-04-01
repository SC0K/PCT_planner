class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 10
    cost_barrier = 50
    sensor_range = 2


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()