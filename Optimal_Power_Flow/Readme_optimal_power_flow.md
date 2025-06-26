# Optimal Power Flow using Pyomo

## Case 1: AC 33 Bus Case (Distribution system)
- 33Bus AC Optimal Power Flow Calculation [Documentation](./0.Tex_file/OPF_basic.pdf)
  - Pandapower case:
    - Basic case: [Pandapower code](./Basic/33_Bus_Case_with_Pandapower/OPF_Case_33bw_panda.ipynb)
    - Pyomo without switch: [Pyomo code](./Distribution/33Bus_pandapower/OPF_Case_33bw_without_switch.py)
    - Pyomo with switch: [Pyomo code](./Distribution/33Bus_pandapower/OPF_Case_33bw_with_switch.py.py)
  - Matpower case:
    - Snapshot 
      - Basic case: [Matpower code](./Basic/33_Bus_Case_with_Matpower/OPF_Case_33bw_matpower.ipynb)
      - Pyomo without switch: [Pyomo code](./Distribution/Matpower/Snapshot/OPF_Case_33bw_without_switch_and_matpower.py)
      - Pyomo with switch: [Pyomo code](./Distribution/Matpower/Snapshot/OPF_Case_33bw_with_switch_and_matpower.py)
    - Multi-period
      - Pyomo without switch: [Pyomo code](./Distribution/Matpower/Multi-period/33Bbus_MPOPF/MPOPF_33Bus_without_switch.py)
