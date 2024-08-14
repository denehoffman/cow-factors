use polars::prelude::*;
use std::{convert::Infallible, env, f64::consts::PI, io::Write, path::PathBuf};

use clap::{CommandFactory, Parser, Subcommand};
use errorfunctions::ComplexErrorFunctions;
use ganesh::prelude::*;
use ganesh::{
    algorithms::{NelderMead, NelderMeadOptions},
    core::Function,
    minimize,
};
use indicatif::ProgressIterator;
use knn::PointCloud;
use num::complex::{Complex, ComplexFloat};
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

fn voigt(x: f64, sigma: f64, gamma: f64) -> f64 {
    let z: Complex<f64> = -Complex::<f64>::I * Complex::new(x, gamma) / (f64::sqrt(2.0) * sigma);
    let wz = z.erfcx();
    wz.re() / (f64::sqrt(2.0 * PI) * sigma)
}
#[derive(Copy, Clone, Debug, PartialEq)]
enum EventType {
    Signal,
    Background,
}

#[derive(Copy, Clone, PartialEq)]
enum PhaseSpaceVariable {
    CosTheta,
    Phi,
    T,
    G,
}

impl PhaseSpaceVariable {
    fn get_d2(&self, a: &Event, b: &Event) -> f64 {
        match self {
            PhaseSpaceVariable::CosTheta => (3.0 / 2.0) * (a.costheta - b.costheta).powi(2),
            PhaseSpaceVariable::Phi => (3.0 / (2.0 * PI.powi(3))) * (a.phi - b.phi).powi(2),
            PhaseSpaceVariable::T => (3.0 / 8.0) * (a.t - b.t).powi(2),
            PhaseSpaceVariable::G => (3.0 / 16.0) * (a.g - b.g).powi(2),
        }
    }
}
fn get_dist_fn(costheta: bool, phi: bool, t: bool, g: bool) -> fn(&Event, &Event) -> f64 {
    match (costheta, phi, t, g) {
        (true, true, true, true) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b)
                    + PhaseSpaceVariable::Phi.get_d2(a, b)
                    + PhaseSpaceVariable::T.get_d2(a, b)
                    + PhaseSpaceVariable::G.get_d2(a, b),
            )
        },
        (true, true, true, false) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b)
                    + PhaseSpaceVariable::Phi.get_d2(a, b)
                    + PhaseSpaceVariable::T.get_d2(a, b),
            )
        },
        (true, true, false, true) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b)
                    + PhaseSpaceVariable::Phi.get_d2(a, b)
                    + PhaseSpaceVariable::G.get_d2(a, b),
            )
        },
        (true, true, false, false) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b) + PhaseSpaceVariable::Phi.get_d2(a, b),
            )
        },
        (true, false, true, true) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b)
                    + PhaseSpaceVariable::T.get_d2(a, b)
                    + PhaseSpaceVariable::G.get_d2(a, b),
            )
        },
        (true, false, true, false) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b) + PhaseSpaceVariable::T.get_d2(a, b),
            )
        },
        (true, false, false, true) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::CosTheta.get_d2(a, b) + PhaseSpaceVariable::G.get_d2(a, b),
            )
        },
        (true, false, false, false) => |a, b| f64::sqrt(PhaseSpaceVariable::CosTheta.get_d2(a, b)),
        (false, true, true, true) => |a, b| {
            f64::sqrt(
                PhaseSpaceVariable::Phi.get_d2(a, b)
                    + PhaseSpaceVariable::T.get_d2(a, b)
                    + PhaseSpaceVariable::G.get_d2(a, b),
            )
        },
        (false, true, true, false) => |a, b| {
            f64::sqrt(PhaseSpaceVariable::Phi.get_d2(a, b) + PhaseSpaceVariable::T.get_d2(a, b))
        },
        (false, true, false, true) => |a, b| {
            f64::sqrt(PhaseSpaceVariable::Phi.get_d2(a, b) + PhaseSpaceVariable::G.get_d2(a, b))
        },
        (false, true, false, false) => |a, b| f64::sqrt(PhaseSpaceVariable::Phi.get_d2(a, b)),
        (false, false, true, true) => |a, b| {
            f64::sqrt(PhaseSpaceVariable::T.get_d2(a, b) + PhaseSpaceVariable::G.get_d2(a, b))
        },
        (false, false, true, false) => |a, b| f64::sqrt(PhaseSpaceVariable::T.get_d2(a, b)),
        (false, false, false, true) => |a, b| f64::sqrt(PhaseSpaceVariable::G.get_d2(a, b)),
        (false, false, false, false) => unimplemented!(),
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Event {
    event_type: EventType,
    mass: f64,
    costheta: f64,
    phi: f64,
    t: f64,
    g: f64,
}
impl Event {
    const M_BOUNDS: (f64, f64) = (0.68, 0.88);
    const T_BOUNDS: (f64, f64) = (0.00, 2.00);
    const G_BOUNDS: (f64, f64) = (-2.00, 2.00);
    const MASS_OMEGA: f64 = 0.78256;
    const GAMMA_OMEGA: f64 = 0.00844;
    const SIGMA_OMEGA: f64 = 0.005;
    const MASS_SLOPE: f64 = 0.3;
    const P_MASS_SIG_MAX: f64 = 50.51;
    const P_MASS_BKG_MAX: f64 = 7.0;
    const RHO_00: f64 = 0.65;
    const RHO_1N1: f64 = 0.05;
    const RHO_10: f64 = 0.10;
    const P_OMEGA_SIG_MAX: f64 = 0.164;
    const P_OMEGA_BKG_MAX: f64 = 1.0 / (3.0 * PI);
    const TAU_SIG: f64 = 0.11;
    const TAU_BKG: f64 = 0.43;
    const P_T_SIG_MAX: f64 = 1.0 / Event::TAU_SIG;
    const P_T_BKG_MAX: f64 = 1.0 / Event::TAU_BKG;
    const G_SIG: f64 = 0.13;
    const G_BKG: f64 = 0.56;
    const P_G_SIG_MAX: f64 = 3.07;
    const P_G_BKG_MAX: f64 = 0.713;
    fn generate<R: Rng>(rng: &mut R, event_type: EventType) -> Self {
        let mass = Event::gen_mass(rng, event_type);
        let (costheta, phi) = Event::gen_omega(rng, event_type);
        let t = Event::gen_t(rng, event_type);
        let g = Event::gen_g(rng, event_type);
        Event {
            event_type,
            mass,
            costheta,
            phi,
            t,
            g,
        }
    }
    fn gen_mass<R: Rng>(rng: &mut R, event_type: EventType) -> f64 {
        let p_max = match event_type {
            EventType::Signal => Event::P_MASS_SIG_MAX,
            EventType::Background => Event::P_MASS_BKG_MAX,
        };
        loop {
            let u_mass = Uniform::new(Event::M_BOUNDS.0, Event::M_BOUNDS.1);
            let u_p_mass = Uniform::new(0.0, p_max);
            let m_star = rng.sample(u_mass);
            if Event::p_mass(m_star, Event::MASS_SLOPE, event_type) >= rng.sample(u_p_mass) {
                return m_star;
            }
        }
    }
    fn gen_omega<R: Rng>(rng: &mut R, event_type: EventType) -> (f64, f64) {
        let p_max = match event_type {
            EventType::Signal => Event::P_OMEGA_SIG_MAX,
            EventType::Background => Event::P_OMEGA_BKG_MAX,
        };
        loop {
            let u_costheta = Uniform::new(-1.0, 1.0);
            let u_phi = Uniform::new(0.0, 2.0 * PI);
            let u_p_omega = Uniform::new(0.0, p_max);
            let costheta_star = rng.sample(u_costheta);
            let phi_star = rng.sample(u_phi);
            if Event::p_omega(
                costheta_star,
                phi_star,
                Event::RHO_00,
                Event::RHO_1N1,
                Event::RHO_10,
                event_type,
            ) >= rng.sample(u_p_omega)
            {
                return (costheta_star, phi_star);
            }
        }
    }

    fn gen_t<R: Rng>(rng: &mut R, event_type: EventType) -> f64 {
        let (p_max, tau) = match event_type {
            EventType::Signal => (Event::P_T_SIG_MAX, Event::TAU_SIG),
            EventType::Background => (Event::P_T_BKG_MAX, Event::TAU_BKG),
        };
        loop {
            let u_t = Uniform::new(Event::T_BOUNDS.0, Event::T_BOUNDS.1);
            let u_p_t = Uniform::new(0.0, p_max);
            let t_star = rng.sample(u_t);
            if Event::p_t(t_star, tau, event_type) >= rng.sample(u_p_t) {
                return t_star;
            }
        }
    }
    fn gen_g<R: Rng>(rng: &mut R, event_type: EventType) -> f64 {
        let (p_max, sigma) = match event_type {
            EventType::Signal => (Event::P_G_SIG_MAX, Event::G_SIG),
            EventType::Background => (Event::P_G_BKG_MAX, Event::G_BKG),
        };
        loop {
            let u_g = Uniform::new(Event::G_BOUNDS.0, Event::G_BOUNDS.1);
            let u_p_g = Uniform::new(0.0, p_max);
            let g_star = rng.sample(u_g);
            if Event::p_g(g_star, sigma, event_type) >= rng.sample(u_p_g) {
                return g_star;
            }
        }
    }
    fn p_mass(mass: f64, m_slope: f64, event_type: EventType) -> f64 {
        match event_type {
            EventType::Signal => voigt(
                mass - Event::MASS_OMEGA,
                Event::SIGMA_OMEGA,
                Event::MASS_OMEGA * Event::GAMMA_OMEGA / 2.0,
            ),
            EventType::Background => {
                2.0 * (Event::M_BOUNDS.0 * (m_slope - 1.0) + Event::M_BOUNDS.1 * m_slope + mass
                    - 2.0 * m_slope * mass)
                    / (Event::M_BOUNDS.1 - Event::M_BOUNDS.0).powi(2)
            }
        }
    }
    fn p_omega(
        costheta: f64,
        phi: f64,
        rho_00: f64,
        rho_1n1: f64,
        rho_10: f64,
        event_type: EventType,
    ) -> f64 {
        let theta = f64::acos(costheta);
        match event_type {
            EventType::Signal => {
                (3.0 / (4.0 * PI))
                    * (0.5 * (1.0 - rho_00) + 0.5 * (3.0 * rho_00 - 1.0) * costheta.powi(2)
                        - rho_1n1 * f64::sin(theta).powi(2) * f64::cos(2.0 * phi)
                        - f64::sqrt(2.0) * rho_10 * f64::sin(2.0 * theta) * f64::cos(phi))
            }
            EventType::Background => (1.0 + f64::abs(f64::sin(theta) * f64::cos(phi))) / (6.0 * PI),
        }
    }
    fn p_t(t: f64, tau: f64, _event_type: EventType) -> f64 {
        f64::exp(-t / tau) / tau
    }
    fn p_g(g: f64, sigma: f64, _event_type: EventType) -> f64 {
        f64::exp(-0.5 * (g / sigma).powi(2)) / (f64::sqrt(2.0 * PI) * sigma)
    }
}

#[derive(Clone)]
struct Dataset {
    events: Vec<Event>,
}

impl Dataset {
    fn len(&self) -> usize {
        self.events.len()
    }
    fn generate(n_signal: usize, n_background: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let signal_events: Vec<Event> = (0..n_signal)
            .map(|_| Event::generate(&mut rng, EventType::Signal))
            .collect();
        let background_events: Vec<Event> = (0..n_background)
            .map(|_| Event::generate(&mut rng, EventType::Background))
            .collect();
        let events = [signal_events, background_events].concat();
        Self { events }
    }
    fn make_pointcloud(&self, ps: Vec<PhaseSpaceVariable>) -> PointCloud<Event> {
        let costheta = ps.contains(&PhaseSpaceVariable::CosTheta);
        let phi = ps.contains(&PhaseSpaceVariable::Phi);
        let t = ps.contains(&PhaseSpaceVariable::T);
        let g = ps.contains(&PhaseSpaceVariable::G);
        let mut pc = PointCloud::new(get_dist_fn(costheta, phi, t, g));
        self.events.iter().for_each(|event| pc.add_point(event));
        pc
    }
    fn knn(&self, index: usize, k: usize, pc: &PointCloud<Event>) -> Dataset {
        Self {
            events: pc
                .get_nearest_k(&self.events[index], k)
                .into_iter()
                .map(|(_, event): (f64, &Event)| *event)
                .collect(),
        }
    }
    fn fit_omega(&self, weights: &Vec<f64>) -> Result<(f64, f64, f64), Infallible> {
        let mut nm = NelderMead::new(
            self.clone(),
            &[0.1, 0.1, 0.1],
            Some(NelderMeadOptions::adaptive(2).simplex_size(0.01).build()),
        );
        nm.minimize(Some(&(FitVariable::Omega, Some(weights))), 1000, |_| {})?;
        let (x, _) = nm.best();
        Ok((x[0], x[1], x[2]))
    }
    fn fit_t(&self, weights: &Vec<f64>) -> Result<f64, Infallible> {
        let mut nm = NelderMead::new(
            self.clone(),
            &[0.2],
            Some(NelderMeadOptions::adaptive(2).simplex_size(0.01).build()),
        );
        nm.minimize(Some(&(FitVariable::T, Some(weights))), 1000, |_| {})?;
        let (x, _) = nm.best();
        Ok(x[0])
    }
    fn fit_g(&self, weights: &Vec<f64>) -> Result<f64, Infallible> {
        let mut nm = NelderMead::new(
            self.clone(),
            &[0.2],
            Some(NelderMeadOptions::adaptive(2).simplex_size(0.01).build()),
        );
        nm.minimize(Some(&(FitVariable::G, Some(weights))), 1000, |_| {})?;
        let (x, _) = nm.best();
        Ok(x[0])
    }
    fn weights(&self, method: Method, weighting: Weighting) -> Result<Vec<f64>, Infallible> {
        match method {
            Method::Standard => weighting.get_weights(self),
            Method::QFactor { k, ps } => {
                let pc = self.make_pointcloud(ps);
                (0..self.len())
                    .into_par_iter()
                    .map(|i: usize| {
                        let knn_event_i = self.knn(i, k, &pc);
                        // assert!(knn_event_i.events[0] == self.events[i]);
                        weighting.get_weights(&knn_event_i).map(|vec| vec[0])
                    })
                    .collect()
            }
        }
    }
    fn analysis(&self, tag: &str, iter: usize, weights: &Vec<f64>) -> Result<String, Infallible> {
        let omega_res = self.fit_omega(weights)?;
        let t_res = self.fit_t(weights)?;
        let g_res = self.fit_g(weights)?;
        Ok(format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tag, iter, omega_res.0, omega_res.1, omega_res.2, t_res, g_res
        ))
    }
}

enum Method {
    Standard,
    QFactor {
        k: usize,
        ps: Vec<PhaseSpaceVariable>,
    },
}

enum Weighting {
    InPlot,
    SPlot,
}

impl Weighting {
    fn get_weights(&self, dataset: &Dataset) -> Result<Vec<f64>, Infallible> {
        match self {
            Weighting::InPlot => Weighting::inplot(dataset),
            Weighting::SPlot => Weighting::splot(dataset),
        }
    }
    fn inplot(dataset: &Dataset) -> Result<Vec<f64>, Infallible> {
        let mut nm = NelderMead::new(
            dataset.clone(),
            &[0.5, 0.5],
            Some(NelderMeadOptions::adaptive(2).simplex_size(0.01).build()),
        );
        minimize!(nm, 1000)?;
        let (x, _) = nm.best();
        // assert!(x[0] < 1.0);
        // assert!(x[0] > 0.0);
        let n_sig = x[0] * dataset.len() as f64;
        let n_bkg = (1.0 - x[0]) * dataset.len() as f64;
        Ok(dataset
            .events
            .par_iter()
            .map(|event: &Event| {
                (n_sig * Event::p_mass(event.mass, x[1], EventType::Signal))
                    / (n_sig * Event::p_mass(event.mass, x[1], EventType::Signal)
                        + n_bkg * Event::p_mass(event.mass, x[1], EventType::Background))
            })
            .collect())
    }
    fn splot(dataset: &Dataset) -> Result<Vec<f64>, Infallible> {
        let mut nm = NelderMead::new(
            dataset.clone(),
            &[0.5, 0.5],
            Some(NelderMeadOptions::adaptive(2).simplex_size(0.01).build()),
        );
        minimize!(nm, 1000)?;
        let (x, _) = nm.best();
        // assert!(x[0] < 1.0);
        // assert!(x[0] > 0.0);
        let n_sig = x[0] * dataset.len() as f64;
        let n_bkg = (1.0 - x[0]) * dataset.len() as f64;
        let v_ss_inv: f64 = dataset
            .events
            .par_iter()
            .map(|event: &Event| {
                Event::p_mass(event.mass, x[1], EventType::Signal).powi(2)
                    / (n_sig * Event::p_mass(event.mass, x[1], EventType::Signal)
                        + n_bkg * Event::p_mass(event.mass, x[1], EventType::Background))
                    .powi(2)
            })
            .sum();
        let v_sb_inv: f64 = dataset
            .events
            .par_iter()
            .map(|event: &Event| {
                Event::p_mass(event.mass, x[1], EventType::Signal)
                    * Event::p_mass(event.mass, x[1], EventType::Background)
                    / (n_sig * Event::p_mass(event.mass, x[1], EventType::Signal)
                        + n_bkg * Event::p_mass(event.mass, x[1], EventType::Background))
                    .powi(2)
            })
            .sum();
        let v_bb_inv: f64 = dataset
            .events
            .par_iter()
            .map(|event: &Event| {
                Event::p_mass(event.mass, x[1], EventType::Background).powi(2)
                    / (n_sig * Event::p_mass(event.mass, x[1], EventType::Signal)
                        + n_bkg * Event::p_mass(event.mass, x[1], EventType::Background))
                    .powi(2)
            })
            .sum();
        let det = v_ss_inv * v_bb_inv - v_sb_inv * v_sb_inv;
        let v_ss = v_bb_inv / det;
        let v_sb = -v_sb_inv / det;
        Ok(dataset
            .events
            .par_iter()
            .map(|event: &Event| {
                (v_ss * Event::p_mass(event.mass, x[1], EventType::Signal)
                    + v_sb * Event::p_mass(event.mass, x[1], EventType::Background))
                    / (n_sig * Event::p_mass(event.mass, x[1], EventType::Signal)
                        + n_bkg * Event::p_mass(event.mass, x[1], EventType::Background))
            })
            .collect())
    }
}

enum FitVariable {
    Mass,
    Omega,
    T,
    G,
}

impl Function<f64, (FitVariable, Option<&Vec<f64>>), Infallible> for Dataset {
    fn evaluate(
        &self,
        x: &[f64],
        args: Option<&(FitVariable, Option<&Vec<f64>>)>,
    ) -> Result<f64, Infallible> {
        let (variable, weights) = if let Some((var, ws_opt)) = args {
            if let Some(ws) = ws_opt {
                (var, *ws)
            } else {
                (var, &vec![1.0; self.len()])
            }
        } else {
            (&FitVariable::Mass, &vec![1.0; self.len()])
        };
        match variable {
            FitVariable::Mass => Ok(-2.0
                * self
                    .events
                    .par_iter()
                    .zip(weights)
                    .map(|(event, weight): (&Event, &f64)| {
                        weight
                            * f64::ln(
                                x[0] * Event::p_mass(event.mass, x[1], EventType::Signal)
                                    + (1.0 - x[0])
                                        * Event::p_mass(event.mass, x[1], EventType::Background),
                            )
                    })
                    .sum::<f64>()),
            FitVariable::Omega => Ok(-2.0
                * self
                    .events
                    .par_iter()
                    .zip(weights)
                    .map(|(event, weight): (&Event, &f64)| {
                        weight
                            * f64::ln(Event::p_omega(
                                event.costheta,
                                event.phi,
                                x[0],
                                x[1],
                                x[2],
                                EventType::Signal,
                            ))
                    })
                    .sum::<f64>()),
            FitVariable::T => Ok(-2.0
                * self
                    .events
                    .par_iter()
                    .zip(weights)
                    .map(|(event, weight): (&Event, &f64)| {
                        weight * f64::ln(Event::p_t(event.t, x[0], EventType::Signal))
                    })
                    .sum::<f64>()),
            FitVariable::G => Ok(-2.0
                * self
                    .events
                    .par_iter()
                    .zip(weights)
                    .map(|(event, weight): (&Event, &f64)| {
                        weight * f64::ln(Event::p_g(event.g, x[0], EventType::Signal))
                    })
                    .sum::<f64>()),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run {
        output: PathBuf,
        #[arg(short, long, value_name = "NSIG", default_value_t = 10000)]
        sig: usize,
        #[arg(short, long, value_name = "NBKG", default_value_t = 10000)]
        bkg: usize,
        #[arg(short, long, value_name = "ITERS", default_value_t = 1)]
        iters: usize,
        #[arg(short, value_name = "K", default_value_t = 100)]
        k: usize,
    },
    Process {
        input: PathBuf,
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Run {
            output,
            sig,
            bkg,
            iters,
            k,
        }) => {
            let mut file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&output)?;
            writeln!(file, "Method\tSeed\tRho00\tRho1n1\tRho10\tTau\tSigma")?;
            writeln!(
                file,
                "Truth\t\t{}\t{}\t{}\t{}\t{}",
                Event::RHO_00,
                Event::RHO_1N1,
                Event::RHO_10,
                Event::TAU_SIG,
                Event::G_SIG
            )?;
            for i in (0..iters).progress() {
                let seed = i as u64;
                let ds = Dataset::generate(sig, bkg, seed);

                let w_none = vec![1.0; ds.len()];
                writeln!(file, "{}", ds.analysis("No Weights", i, &w_none)?)?;

                let w_inplot = ds.weights(Method::Standard, Weighting::InPlot)?;
                writeln!(file, "{}", ds.analysis("inPlot", i, &w_inplot)?)?;

                let w_qfactors_ct = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::CosTheta],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("Q-Factors (CosTheta)", i, &w_qfactors_ct)?
                )?;

                let w_qfactors_phi = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::Phi],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("Q-Factors (Phi)", i, &w_qfactors_phi)?
                )?;

                let w_qfactors_t = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::T],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(file, "{}", ds.analysis("Q-Factors (T)", i, &w_qfactors_t)?)?;

                let w_qfactors_g = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::G],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(file, "{}", ds.analysis("Q-Factors (G)", i, &w_qfactors_g)?)?;

                let w_qfactors_ct_phi = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::CosTheta, PhaseSpaceVariable::Phi],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("Q-Factors (CosTheta, Phi)", i, &w_qfactors_ct_phi)?
                )?;

                let w_qfactors_ct_phi_t = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![
                            PhaseSpaceVariable::CosTheta,
                            PhaseSpaceVariable::Phi,
                            PhaseSpaceVariable::T,
                        ],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("Q-Factors (CosTheta, Phi, T)", i, &w_qfactors_ct_phi_t,)?
                )?;

                let w_qfactors_ct_phi_g = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![
                            PhaseSpaceVariable::CosTheta,
                            PhaseSpaceVariable::Phi,
                            PhaseSpaceVariable::G,
                        ],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("Q-Factors (CosTheta, Phi, G)", i, &w_qfactors_ct_phi_g,)?
                )?;

                let w_qfactors_ct_phi_t_g = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![
                            PhaseSpaceVariable::CosTheta,
                            PhaseSpaceVariable::Phi,
                            PhaseSpaceVariable::T,
                            PhaseSpaceVariable::G,
                        ],
                    },
                    Weighting::InPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("Q-Factors (CosTheta, Phi, T, G)", i, &w_qfactors_ct_phi_t_g,)?
                )?;

                let w_splot = ds.weights(Method::Standard, Weighting::SPlot)?;
                writeln!(file, "{}", ds.analysis("sPlot", i, &w_splot)?)?;

                let w_sqfactors_ct = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::CosTheta],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (CosTheta)", i, &w_sqfactors_ct)?
                )?;

                let w_sqfactors_phi = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::Phi],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (Phi)", i, &w_sqfactors_phi)?
                )?;

                let w_sqfactors_t = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::T],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (T)", i, &w_sqfactors_t)?
                )?;

                let w_sqfactors_g = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::G],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (G)", i, &w_sqfactors_g)?
                )?;

                let w_sqfactors_ct_phi = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![PhaseSpaceVariable::CosTheta, PhaseSpaceVariable::Phi],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (CosTheta, Phi)", i, &w_sqfactors_ct_phi)?
                )?;

                let w_sqfactors_ct_phi_t = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![
                            PhaseSpaceVariable::CosTheta,
                            PhaseSpaceVariable::Phi,
                            PhaseSpaceVariable::T,
                        ],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (CosTheta, Phi, T)", i, &w_sqfactors_ct_phi_t,)?
                )?;

                let w_sqfactors_ct_phi_g = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![
                            PhaseSpaceVariable::CosTheta,
                            PhaseSpaceVariable::Phi,
                            PhaseSpaceVariable::G,
                        ],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis("sQ-Factors (CosTheta, Phi, G)", i, &w_sqfactors_ct_phi_g,)?
                )?;

                let w_sqfactors_ct_phi_t_g = ds.weights(
                    Method::QFactor {
                        k,
                        ps: vec![
                            PhaseSpaceVariable::CosTheta,
                            PhaseSpaceVariable::Phi,
                            PhaseSpaceVariable::T,
                            PhaseSpaceVariable::G,
                        ],
                    },
                    Weighting::SPlot,
                )?;
                writeln!(
                    file,
                    "{}",
                    ds.analysis(
                        "sQ-Factors (CosTheta, Phi, T, G)",
                        i,
                        &w_sqfactors_ct_phi_t_g,
                    )?
                )?;
            }
        }
        Some(Commands::Process { input, output }) => {
            let mut df = CsvReadOptions::default()
                .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
                .with_has_header(true)
                .try_into_reader_with_file_path(Some(input))?
                .finish()?;
            let truth = df.slice(0, 1);
            let truth = truth.get(0).unwrap_or(vec![
                "".into(),
                0.into(),
                Event::RHO_00.into(),
                Event::RHO_1N1.into(),
                Event::RHO_10.into(),
                Event::TAU_SIG.into(),
                Event::G_SIG.into(),
            ]);
            df = df.slice(1, df.height() - 1);
            df = df
                .lazy()
                .with_columns([
                    lit(truth[2].try_extract::<f64>()?).alias("Rho00_truth"),
                    lit(truth[3].try_extract::<f64>()?).alias("Rho1n1_truth"),
                    lit(truth[4].try_extract::<f64>()?).alias("Rho10_truth"),
                    lit(truth[5].try_extract::<f64>()?).alias("Tau_truth"),
                    lit(truth[6].try_extract::<f64>()?).alias("Sigma_truth"),
                ])
                .collect()?;
            let standard_devs = df
                .clone()
                .lazy()
                .group_by_stable([col("Method")])
                .agg([
                    col("Rho00").std(1).alias("Rho00_std"),
                    col("Rho1n1").std(1).alias("Rho1n1_std"),
                    col("Rho10").std(1).alias("Rho10_std"),
                    col("Tau").std(1).alias("Tau_std"),
                    col("Sigma").std(1).alias("Sigma_std"),
                ])
                .collect()?;
            let merged = df
                .lazy()
                .join(
                    standard_devs.lazy(),
                    [col("Method")],
                    [col("Method")],
                    JoinArgs::default(),
                )
                .with_columns([
                    ((col("Rho00") - col("Rho00_truth")) / col("Rho00_std")).alias("Rho00_pull"),
                    ((col("Rho1n1") - col("Rho1n1_truth")) / col("Rho1n1_std"))
                        .alias("Rho1n1_pull"),
                    ((col("Rho10") - col("Rho10_truth")) / col("Rho10_std")).alias("Rho10_pull"),
                    ((col("Tau") - col("Tau_truth")) / col("Tau_std")).alias("Tau_pull"),
                    ((col("Sigma") - col("Sigma_truth")) / col("Sigma_std")).alias("Sigma_pull"),
                ])
                .collect()?;
            let mut res = merged
                .lazy()
                .group_by_stable([col("Method")])
                .agg([
                    col("Rho00_pull").mean().alias("Rho00_mean_pull"),
                    col("Rho1n1_pull").mean().alias("Rho1n1_mean_pull"),
                    col("Rho10_pull").mean().alias("Rho10_mean_pull"),
                    col("Tau_pull").mean().alias("Tau_mean_pull"),
                    col("Sigma_pull").mean().alias("Sigma_mean_pull"),
                ])
                .collect()?;
            env::set_var("POLARS_FMT_TABLE_ROUNDED_CORNERS", "1");
            env::set_var("POLARS_FMT_MAX_COLS", "-1");
            env::set_var("POLARS_FMT_MAX_ROWS", "-1");
            env::set_var("POLARS_FMT_STR_LEN", "50");
            println!("{}", res);
            if let Some(out_path) = output {
                println!("Writing result to {:?}", out_path);
                let file = std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&out_path)?;
                let mut writer = CsvWriter::new(file);
                writer = writer.with_separator(b'\t');
                writer.finish(&mut res)?;
            }
        }
        None => Cli::command().print_help()?,
    }
    Ok(())
}
