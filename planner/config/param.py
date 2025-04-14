class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 10
    cost_barrier = 50
    coverage_threshold = 0.6


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'
class ConfigSensor():
    sensor_range = 2.0
    sensor_fov = 360    # degrees


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()
    sensor = ConfigSensor()