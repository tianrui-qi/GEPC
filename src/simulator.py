import ast
import copy
import importlib.util
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp


if importlib.util.find_spec("gillespy2") is None:
    gp2 = None
    warnings.warn("Could not load GillesPy2 module.")
else:
    import gillespy2 as gp2


__all__ = [
    "CcaSR_gillespie",
    "CcaSR_gillespie_simple_noE",
    "CcaSR_gillespie_simple",
    "camera_sim",
    "simulate",
]


@contextmanager
def _numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class Reaction:
    def __init__(self, propensity: str, stoichiometry: dict[str, int]) -> None:
        self.propensity = propensity
        self._compiled = compile(propensity, "_", "eval")
        self.stoichiometry = stoichiometry

    def update(self, params: dict, species: dict) -> float:
        return eval(self._compiled, {}, {**params, **species})

    def serialize(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }


class PowForDoubleStar(ast.NodeTransformer):
    pow_func = ast.parse("pow", mode="eval").body

    def visit_BinOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            node = ast.copy_location(
                ast.Call(
                    func=self.pow_func,
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node

    def mytransform(self, expression: str) -> str:
        tree = ast.parse(expression, mode="eval")
        tree = self.visit(tree)
        return ast.unparse(tree)


class CcaSR_gillespie:
    def __init__(
        self,
        sampling: int,
        eta: float = 1.0,
        nu: float = 0.01,
        h1: float = 4e-2,
        h2: float = 1e-3,
        a: float = 0.025,
        k_h: float = 45.0,
        n_h: float = 3.6,
        tau: float = 12.0,
        **kwargs,
    ) -> None:
        self._init_kwargs = {
            "sampling": sampling,
            "eta": eta,
            "nu": nu,
            "h1": h1,
            "h2": h2,
            "a": a,
            "k_h": k_h,
            "n_h": n_h,
            "tau": tau,
        }
        self._gp2model = None
        self._odemodel = None
        self.params = {
            "eta": eta,
            "nu": nu,
            "h1": h1,
            "h2": h2,
            "a": a,
            "K_H": k_h,
            "nh": n_h,
            "tau": tau,
        }
        self.species = {
            "U": 0,
            "H": 0.0,
            "E": round(np.random.poisson(self.params["h1"] / self.params["h2"])),
            "F": 0,
        }
        self.reactions = (
            Reaction("h1", {"E": 1}),
            Reaction("h2*E", {"E": -1}),
            Reaction("eta*U", {"H": 1}),
            Reaction("nu*H", {"H": -1}),
            Reaction("a*E* H**nh/(K_H**nh + H**nh)", {"F": 1}),
            Reaction("nu*F", {"F": -1}),
        )
        self.sampling = sampling
        self.events = []
        self.past_events = []
        self.time = 0

    def _compile_gp2_model(self):
        if gp2 is None:
            raise ImportError("gillespy2 is required when solver='gillespy2'.")

        model = gp2.Model(name=self.__class__.__name__)
        for name, value in self.params.items():
            model.add_parameter(gp2.Parameter(name=name, expression=value))
        for name, value in self.species.items():
            species = gp2.Species(name=name, initial_value=value)
            species.mode = "discrete"
            model.add_species(species)

        transformer = PowForDoubleStar()
        for index, reaction in enumerate(self.reactions):
            propensity = transformer.mytransform(reaction.propensity)
            model.add_reaction(
                gp2.Reaction(
                    name=f"reaction_{index}",
                    propensity_function=propensity,
                    ode_propensity_function=propensity,
                    reactants={
                        name: -value
                        for name, value in reaction.stoichiometry.items()
                        if value < 0
                    },
                    products={
                        name: value
                        for name, value in reaction.stoichiometry.items()
                        if value > 0
                    },
                )
            )
        return model

    def _compile_ode_model(self) -> tuple[dict[str, str], dict[str, object]]:
        system = {name: "" for name in self.species}
        for reaction in self.reactions:
            for name, value in reaction.stoichiometry.items():
                sign = "+" if value > 0 else "-"
                system[name] += f"{sign}{abs(value)}*{reaction.propensity} "

        compiled = {}
        for name, expression in system.items():
            if len(expression) == 0:
                expression = "0"
                system[name] = expression
            compiled[name] = compile(expression, f"d{name}_dt", "eval")
        return system, compiled

    def _ode_computation(self, t, y):
        species = {key: y[index] for index, key in enumerate(self.species)}
        dydt = np.empty_like(y)
        for index, key in enumerate(self._odemodel):
            dydt[index] = eval(self._odemodel[key], {}, {**self.params, **species})
        return dydt

    def run(self, stoptime: float, realizations: int = 1, solver: str = "original"):
        self.events.sort(key=lambda elem: elem["time"])
        solver = solver.lower()
        if solver in ("original", "og"):
            return self._run_original(stoptime, realizations)
        if solver in ("gillespy2", "gp2"):
            return self._run_gp2(stoptime, realizations)
        if solver == "ode":
            return self._run_ode(stoptime)
        raise ValueError(f"Unknown solver: {solver}")

    def _run_original(self, stoptime: float, realizations: int = 1):
        if realizations > 1:
            time_series = []
            for _ in range(realizations):
                realize = self.copy()
                time_series.append(realize._run_original(stoptime))
            self.copy(realize)
            return time_series

        starttime = self.time
        time_series = []
        newspecies = self.species.copy()

        while self.time < stoptime:
            self.species = newspecies
            timestep, newspecies = self._run_nextreaction()
            self.time += timestep

            while (self.time - starttime) >= len(time_series) * self.sampling:
                time_series.append(self.species.copy())

        self.time = stoptime

        for event in self.past_events[::-1]:
            if event["time"] >= self.time:
                self.events.insert(0, event)
                self.past_events.remove(event)

        return time_series[: int((stoptime - starttime) / self.sampling) + 1]

    def _run_gp2(self, stoptime: float, realizations: int = 1):
        if self._gp2model is None:
            self._gp2model = self._compile_gp2_model()

        for name, value in self.species.items():
            self._gp2model.get_species(name).set_initial_value(value)

        remove_me = []
        for event_index, event in enumerate(self.events):
            if event["time"] < stoptime:
                assignments = [
                    gp2.EventAssignment(variable=name, expression=value)
                    for name, value in event["set"]
                ]
                self._gp2model.add_event(
                    gp2.Event(
                        name=f"event_{event_index}",
                        assignments=assignments,
                        trigger=gp2.EventTrigger("1", initial_value=False),
                        delay=f"{event['time'] - self.time}",
                    )
                )
                remove_me.append(event)
                self.past_events.append(event)
        for event in remove_me:
            self.events.remove(event)

        self._gp2model.timespan(
            gp2.TimeSpan.arange(increment=self.sampling, t=stoptime - self.time)
        )
        results = self._gp2model.run(
            number_of_trajectories=realizations,
            algorithm="Tau-Hybrid",
        )
        self._gp2model.delete_all_events()

        timeseries = []
        for res in results:
            timeseries.append([])
            for t in range(len(res[tuple(self.species.keys())[0]])):
                timeseries[-1].append({name: res[name][t] for name in self.species})

        self.time = stoptime
        for name in self.species:
            self.species[name] = timeseries[-1][-1][name]

        if realizations == 1:
            return timeseries[0]
        return timeseries

    def _run_ode(self, stoptime: float):
        if self._odemodel is None:
            _, self._odemodel = self._compile_ode_model()

        sampling_times = np.arange(0, stoptime + 1, self.sampling)
        y0 = np.array([value for value in self.species.values()], dtype=np.float32)
        record = [y0[:, np.newaxis]]

        while self.time < stoptime:
            next_event = self.events[0] if self.events else {"time": stoptime + 1, "set": ()}
            if next_event["time"] <= self.time:
                for name, value in next_event["set"]:
                    y0[tuple(self.species.keys()).index(name)] = value
                self.events.remove(next_event)
                continue

            run_to = min(next_event["time"], stoptime)
            sampling = np.logical_and(sampling_times > self.time, sampling_times <= run_to)
            sampling = [
                sampling_times[index]
                for index, should_sample in enumerate(sampling)
                if should_sample
            ]
            if run_to != stoptime:
                sampling.append(run_to)

            if sampling:
                y = solve_ivp(
                    self._ode_computation,
                    t_span=(0, run_to - self.time),
                    y0=y0,
                    t_eval=[sample - self.time for sample in sampling],
                    method="LSODA",
                )
                y_arr = np.asarray(y.y, dtype=np.float32)
                if y_arr.size == 0:
                    y_arr = y0[:, np.newaxis]
                if y_arr.ndim == 1:
                    y_arr = y_arr[:, np.newaxis]
                y0 = y_arr[:, -1]
                record.append(y_arr[:, :-1] if run_to != stoptime else y_arr)

            self.time = run_to
            if next_event["time"] <= self.time and next_event in self.events:
                for name, value in next_event["set"]:
                    y0[tuple(self.species.keys()).index(name)] = value
                self.events.remove(next_event)

        for index, name in enumerate(self.species):
            self.species[name] = y0[index]

        record = np.concatenate(record, axis=1)
        return [
            {name: record[index, t] for index, name in enumerate(self.species)}
            for t in range(record.shape[1])
        ]

    def set_light_events(self, light_sequence) -> None:
        for index, value in enumerate(light_sequence):
            self.events.append(
                {
                    "time": self.time + index * self.sampling + self.params["tau"],
                    "set": (["U", value],),
                }
            )

    def _run_nextreaction(self):
        propensities = [reaction.update(self.params, self.species) for reaction in self.reactions]
        cumulative = np.cumsum(propensities)
        timestep = -np.log(np.random.random()) / cumulative[-1]
        newspecies = self.species.copy()

        if len(self.events) > 0 and self.time + timestep > self.events[0]["time"]:
            event = self.events.pop(0)
            self.past_events.append(event)
            if "set" in event:
                for name, value in event["set"]:
                    newspecies[name] = value
            elif "react" in event:
                for name, value in event["react"]:
                    newspecies[name] = self.species[name] + value
            timestep = event["time"] - self.time
        else:
            selected = np.argmax(cumulative >= np.random.random() * cumulative[-1])
            for name, value in self.reactions[selected].stoichiometry.items():
                newspecies[name] = self.species[name] + value

        return timestep, newspecies

    def serialize(self) -> dict:
        serial = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
        serial["reactions"] = tuple(reaction.serialize() for reaction in serial["reactions"])
        return copy.deepcopy(serial)

    def load(self, serial: dict) -> None:
        for key, value in copy.deepcopy(serial).items():
            setattr(self, key, value)
        self.reactions = tuple(
            Reaction(reaction["propensity"], reaction["stoichiometry"])
            for reaction in self.reactions
        )

    def copy(self, copy_from=None):
        if copy_from is None:
            copy_to = self.__class__(**self._init_kwargs)
            copy_from = self
        else:
            if copy_from.__class__ != self.__class__:
                raise ValueError("You can only copy from the same cell class.")
            copy_to = self
        copy_to.load(copy_from.serialize())
        return copy_to

    def decompile(self) -> None:
        self._odemodel = None
        self._gp2model = None

    def _resample_species(self) -> None:
        self.species = {
            "U": 0,
            "H": 0.0,
            "E": round(np.random.poisson(self.params["h1"] / self.params["h2"])),
            "F": 0,
        }

    def update_params(self, new_params: dict) -> None:
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value
            elif key in self.species:
                self.species[key] = value
            elif key == "resample_species":
                self._resample_species()
            else:
                print("Suggested new parameter is not in either species or parameter list.")


class CcaSR_gillespie_simple_noE(CcaSR_gillespie):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.species = {
            "U": 0,
            "H": 0.0,
            "E": round(self.params["h1"] / self.params["h2"]),
            "F": 0,
        }
        self.reactions = (
            Reaction("eta*U", {"H": 1}),
            Reaction("nu*H", {"H": -1}),
            Reaction("a*E* H**nh/(K_H**nh + H**nh)", {"F": 1}),
            Reaction("nu*F", {"F": -1}),
        )


class CcaSR_gillespie_simple(CcaSR_gillespie):
    def __init__(
        self,
        mu: float = 40.0,
        sigma: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._init_kwargs = {**self._init_kwargs, "mu": mu, "sigma": sigma}
        self.params = {
            "eta": self.params["eta"],
            "nu": self.params["nu"],
            "mu": mu,
            "sigma": sigma,
            "a": self.params["a"],
            "K_H": self.params["K_H"],
            "nh": self.params["nh"],
            "tau": self.params["tau"],
        }
        self.species = {
            "U": 0,
            "H": 0.0,
            "E": round(np.random.normal(loc=self.params["mu"], scale=self.params["sigma"])),
            "F": 0,
        }
        self.reactions = (
            Reaction("eta*U", {"H": 1}),
            Reaction("nu*H", {"H": -1}),
            Reaction("a*E* H**nh/(K_H**nh + H**nh)", {"F": 1}),
            Reaction("nu*F", {"F": -1}),
        )

    def _resample_species(self) -> None:
        self.species = {
            "U": 0,
            "H": 0.0,
            "E": round(np.random.normal(loc=self.params["mu"], scale=self.params["sigma"])),
            "F": 0,
        }


def camera_sim(
    fluo,
    camera_mult: float = 40,
    camera_max: float = 4095,
    camera_offset: float = 100,
    noise_perc: float = 5,
):
    return np.clip(
        (fluo * camera_mult + camera_offset)
        * np.random.normal(loc=1.0, scale=noise_perc / 100, size=fluo.shape),
        0,
        camera_max,
    )


def simulate(
    cell_class: str,
    cell_seed: int,
    stimulus: np.ndarray,
    solver: str,
    camera_noise_perc: float,
    sim_seed: int,
    measure_seed: int,
    sampling: int,
    camera_mult: float = 40.0,
    camera_offset: float = 100.0,
    camera_max: float = 4095.0,
    eta: float = 1.0,
    nu: float = 0.01,
    h1: float = 4e-2,
    h2: float = 1e-3,
    a: float = 0.025,
    k_h: float = 45.0,
    n_h: float = 3.6,
    tau: float = 12.0,
    mu: float = 40.0,
    sigma: float = 2.0,
):
    cell_type = globals()[cell_class]
    cell_kwargs = {
        "sampling": sampling,
        "eta": eta,
        "nu": nu,
        "h1": h1,
        "h2": h2,
        "a": a,
        "k_h": k_h,
        "n_h": n_h,
        "tau": tau,
        "mu": mu,
        "sigma": sigma,
    }

    with _numpy_seed(cell_seed):
        cell = cell_type(**cell_kwargs)
    serial = cell.serialize()

    with _numpy_seed(sim_seed):
        cell = cell_type(**cell_kwargs)
        cell.load(copy.deepcopy(serial))
        cell.set_light_events(stimulus.astype(float))
        timeseries = cell.run(
            cell.time + len(stimulus) * int(cell.sampling),
            solver=solver,
        )
        fluo = np.array(
            [point["F"] for point in timeseries[1:]],
            dtype=np.float32,
        )[-len(stimulus):]

    with _numpy_seed(measure_seed):
        observed = camera_sim(
            fluo,
            camera_mult=camera_mult,
            camera_max=camera_max,
            camera_offset=camera_offset,
            noise_perc=float(camera_noise_perc),
        )

    return SimpleNamespace(
        observed=observed.astype(np.float32),
        stimulus=stimulus.astype(np.float32),
        cell_class=cell_class,
        solver=solver,
        sampling=sampling,
        camera_mult=camera_mult,
        camera_offset=camera_offset,
        camera_max=camera_max,
        camera_noise_perc=camera_noise_perc,
        eta=eta,
        nu=nu,
        h1=h1,
        h2=h2,
        a=a,
        k_h=k_h,
        n_h=n_h,
        tau=tau,
        mu=mu,
        sigma=sigma,
        cell_seed=cell_seed,
        sim_seed=sim_seed,
        measure_seed=measure_seed,
        cell_E=float(serial.get("species", {}).get("E", 40.0)),
    )
