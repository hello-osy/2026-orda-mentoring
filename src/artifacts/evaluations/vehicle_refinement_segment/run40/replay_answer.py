import sys
from orda_sim import answer
OriginalDriver=answer.Driver
class ReplayDriver(OriginalDriver):
    def __init__(self, config):
        super().__init__(config)
        # Test-only seed from full13 sensor-memory telemetry, not a map command.
        self.stage_index=config['answer']['missions']['sequence'].index('traffic')
        self.road_memory=True
        self.road_heading=-.0015675340633240302
        self.road_center=-.325510309001805
        self.avoiding=True
        self.avoidance_previous=.0325
        self.fork_remaining=4.678225326779479
        self.fork_bias=0.
        self.fork_pending=True
        self.fork_target=-.55
answer.Driver=ReplayDriver
answer.main(sys.argv[1])
