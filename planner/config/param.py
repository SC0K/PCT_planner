class ConfigPlanner():
    use_quintic = False
    max_heading_rate = 30
    cost_barrier = 50
    coverage_threshold = 0.95


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'
class ConfigSensor():
    sensor_range = 3.5
    sensor_fov = 90    # degrees
    sensor_fov_ver = 90
    sensor_fov_hor = 90    # degrees


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()
    sensor = ConfigSensor()