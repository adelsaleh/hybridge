"""Human-readable case and active-integrator labels for guiding-center output."""


def run_label(config):
    """Describe the resolved case and scheme without relying on preset filenames."""
    parameters = config.case_params
    if config.case == "euler_vortex_gas":
        model = "Euler vortex-gas turbulence"
    elif config.case == "positive_turbulence":
        model = "Positive guiding-center turbulence"
        geometry = parameters.get("geometry", "disc")
        if geometry != "disc":
            geometry = {"iter": "ITER", "horseshoe": "Horseshoe", "pacman": "Pac-Man",
                        "smooth-star": "star with island"}.get(geometry, geometry)
            model += f" ({geometry})"
    elif config.case == "euler_shaped_vortex_gas":
        geometry = parameters.get("geometry", "horseshoe")
        geometry = {"horseshoe": "Horseshoe", "iter": "ITER", "pacman": "Pac-Man"}.get(geometry, geometry)
        model = f"Euler vortex gas ({geometry})"
    elif config.case == "diocotron_k":
        model = f"Diocotron m={parameters.get('k', 9)}"
        if parameters.get("p") == 2:
            model += " (Gaussian annulus)"
    else:
        model = config.case.replace("_", " ")
    scheme = {
        "imex-ark3": "IMEX-ARK3", "si-euler": "SI Euler",
        "predictor-corrector": "Predictor-corrector", "si-bdf2": "SI BDF2", "si-bdf3": "SI BDF3",
        "h1-bdf3": "H1-BDF3", "h2-bdf3": "H2-BDF3",
    }.get(config.time_scheme, config.time_scheme)
    if getattr(config, "density_positivity", "none") == "kkt":
        scheme += " + KKT positivity"
    return f"{model} | {scheme}"
