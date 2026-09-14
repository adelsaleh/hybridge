# Campagne complète de performance ADR

Cette campagne complète `run_adv_diff_rea_campaign.py` et conserve sa première
validation. Le nouveau point d'entrée est :

```bash
PYTHONPATH=. python scripts/run_adv_diff_rea_performance.py --help
```

L'objectif principal est le **temps de préparation neuve du solveur + résolution
GMRES jusqu'à convergence**, en float64 avec un résidu vrai à `rtol=1e-12` par
défaut. L'assemblage est partagé entre candidats afin que tous résolvent le même
système. Les meilleurs candidats sont aussi mesurés de bout en bout, avec un
nouvel assemblage pour chaque répétition.

## Trois niveaux

| Niveau | Périmètre |
|---|---|
| `smoke` | Les cinq cas manufacturés, maillages 4 et 8, p=2, un chemin pour chacune des six familles, référence CPU directe et diagnostics. Pas de grand système. |
| `full` | Tous les chemins valides sur les problèmes de sélection, réglage des paramètres de chemins présélectionnés, puis chaque chemin sur des problèmes proches d'un million de dofs. |
| `exhaustive` | Produit cartésien complet des chemins et des paramètres sélectionnés, sur les problèmes de sélection **et** les grands systèmes. Beaucoup plus coûteux. |

`full` est une recherche par étapes : elle n'affirme pas trouver l'optimum global
de tout le produit cartésien. Pour chaque cas et chaque ordre, elle :

1. vérifie le cas sur deux petits maillages avec une référence CPU indépendante ;
2. compare les 108 chemins valides avec les paramètres de départ indiqués dans le manifeste ;
3. retient, par défaut, le meilleur chemin de chaque famille pour explorer la grille de paramètres ;
4. reprend la meilleure configuration connue **de chaque chemin** sur le grand système ;
5. profile séparément le meilleur candidat convergé de chaque chemin ;
6. remesure de bout en bout le meilleur candidat convergé de chaque famille.

Un chemin qui échoue lors de la sélection est encore tenté sur le grand système,
avec ses paramètres de départ, lorsque `--large-scope paths` (défaut) est actif.
`--large-scope finalists` permet de limiter les grands essais aux gagnants par
famille, mais réduit explicitement la couverture. `--tuning-paths-per-family 2`
permet d'explorer les paramètres des deux meilleurs chemins par famille.

## Chemins et paramètres réellement couverts

| Choix | Valeurs disponibles et explorées par défaut dans `full` |
|---|---|
| Famille | none, poly, block_jacobi, block_jacobi_poly, asm, asm_poly |
| Matvec face-dense | raw, raw_fused, matmul |
| Application BJ | raw, matmul |
| Application ASM | raw, matmul, fused |
| Solveur local du préconditionneur | cpu_inverse, gpu_inverse, cublas_inverse, gpu_solve |
| Degré polynomial | 6, 12, 18, 24, pendant le réglage |
| Restart | 50, 75, 100, pendant le réglage |
| Orthogonalisation de GMRES | mgs, mgs2, cgs, cgs2, pendant le réglage |
| Orthogonalisation du setup polynomial | mgs, mgs2, cgs, cgs2, pendant le réglage |

Les produits cartésiens inutiles sont supprimés : degré polynomial absent sans
polynôme, solveur local absent sans BJ/ASM, etc. `gpu_solve` ne supporte que
`matmul` : les associations avec `raw` ou `fused` sont exclues, pas comptées comme
des essais. `external_inverse` est un mode d'injection d'inverses fournis par
l'appelant, et non une méthode de construction autonome : il n'est pas un chemin
supplémentaire de cette campagne.

Le repli automatique CGS vers CGS2 est désactivé pour que la comparaison porte
sur l'orthogonalisation demandée. Chaque essai garde le contrôle du résidu vrai.
L'autotuning opaque est désactivé : les paramètres exécutés sont explicitement
inscrits dans le manifeste et les résultats.

L'assemblage ADR lui-même utilise un backend choisi pour la campagne
(`--assembly-backend cupy` par défaut, ou `numpy`). Les quatre solveurs locaux
ci-dessus concernent les **préconditionneurs**, pas l'inversion HDG `A_T^{-1}`,
qui reste `cupy.linalg.inv` dans le backend CuPy.

## Cas et grandes tailles

Les cinq cas sont les solutions manufacturées de `adv_diff_rea_cases.py` :
quadratic, variable_velocity, trigonometric, advection_dominated et anisotropic.
Cette campagne ne transforme pas ces cas lisses en simulations réalistes de
couches limites ou de plasmas. La provenance des cas et les coefficients sont
décrits dans `adv_diff_rea_validation.md`.

Les ordres par défaut sont p=3 et p=4. Le maillage de sélection est 16×16. Le
maillage final est calculé automatiquement pour s'approcher au mieux de la cible
`--target-dofs`, après élimination de Dirichlet :

\[
N_{\mathrm{trace,libres}}=(3n^2-2n)(p+1).
\]

| Ordre | Maillage automatique pour 1 000 000 | Dofs libres |
|---|---|---:|
| p=3 | 289×289 | 999 940 |
| p=4 | 259×259 | 1 003 625 |

On peut ajouter p=6 avec `--orders 3 4 6`. Le fichier `manifest.json` donne les
estimations de mémoire hôte, mémoire GPU et disque pour chaque taille. Pour les
deux tailles par défaut, l'estimation conservatrice d'assemblage GPU est
respectivement d'environ 7,9 et 12,9 Gio, et celle de RAM hôte 13,6 et 21,4 Gio.
Ces valeurs ne sont **ni des pics mesurés ni une garantie de tenue en mémoire**.
Les allocations de CuPy, les intermédiaires de quadrature et les autres processus
peuvent modifier la capacité disponible.

Chaque worker contrôle les ressources disponibles et le budget
`--memory-fraction` (0,8 par défaut). `--host-limit-gib` ajoute une limite hôte
explicite. Un manque de mémoire estimé apparaît comme `skipped_budget` et rend
la campagne incomplète si une étape requise ne peut pas être exécutée. Les caches
sur disque conservent notamment les inverses locales nécessaires à la
reconstruction ; les grands cas peuvent occuper plusieurs dizaines de Gio au
total. Le manifeste chiffre les tailles estimées. Ne pas démarrer par le niveau
exhaustive pour découvrir la mémoire réellement nécessaire.

## Commandes à lancer sur la machine GPU

```bash
PYTHONPATH=. python scripts/run_adv_diff_rea_performance.py \
  --level full --dry-run --output results/adr_full
```

Vérifier d'abord les nouvelles connexions numériques et le profilage :

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_adv_diff_rea.py tests/test_adv_diff_rea_performance.py

OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/run_adv_diff_rea_performance.py \
  --level smoke --output results/adr_smoke
```

Puis lancer la campagne complète :

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/run_adv_diff_rea_performance.py \
  --level full --output results/adr_full
```

La campagne complète peut être longue : avec les défauts, elle prévoit dix
couples sélection/grand système, 108 chemins par couple et jusqu'à 612 candidats
de réglage par problème (certains doublonnent les essais de départ et sont
réutilisés). Chaque configuration est échauffée deux fois puis mesurée cinq
fois. La grille exhaustive comprend **11 016 configurations par problème**
avec les paramètres par défaut. Ces nombres sont des essais planifiés, pas des
résultats exécutés ici.

Exemple de réduction explicite du coût tout en gardant les 108 chemins :

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/run_adv_diff_rea_performance.py \
  --level full --cases trigonometric advection_dominated anisotropic \
  --orders 3 4 --degrees 12 18 --restarts 50 100 \
  --orthogonalizations cgs cgs2 --polynomial-orthogonalizations cgs2 \
  --warmup 1 --repeats 3 --output results/adr_full_reduced
```

Reprendre avec **les mêmes paramètres** et ajouter `--resume`. Pour retenter
également les erreurs terminées ou les essais omis pour budget, ajouter
`--retry-failed`. Un essai interrompu en cours est relancé ; ses quelques
répétitions ne sont jamais traitées comme une configuration entièrement mesurée.
La reprise vérifie les paramètres, l'empreinte du code et l'environnement matériel
et logiciel. Utiliser un autre répertoire pour une autre carte ou une autre
configuration de campagne.

Le délai `--timeout` (1800 s par défaut) porte sur chaque worker de mesure ou de
profilage, toutes répétitions incluses. `--assembly-timeout` (3600 s) s'applique
à l'assemblage et aux confirmations de bout en bout. Les travailleurs sont des
processus distincts, exécutés séquentiellement sur un GPU. Le délai protège
l'orchestrateur et permet de récupérer le contexte CUDA à la fin du processus.

## Mesures et critères de classement

Chaque répétition chronométrée reconstruit l'opérateur, le préconditionneur et
l'espace de travail GMRES, puis résout avec x0=0. L'échauffement peut mettre en
cache les noyaux compilés ; les temps classés représentent un environnement déjà
échauffé avec une préparation algébrique neuve. Le processus Python, les imports
et la lecture du cache sont hors du temps de résolution.

Les temps primaires sont muraux, synchronisés sur le flux CUDA :

- `operator_setup_ms` : construction et transfert de l'opérateur face-dense ;
- `base_preconditioner_setup_ms` : construction de BJ ou ASM et factorisation/inversion ;
- `polynomial_setup_ms` : Arnoldi spectral, Ritz harmoniques, Leja et setup du polynôme ;
- `gmres_workspace_setup_ms` et `rhs_transfer_ms` ;
- `setup_ms`, `solve_ms`, **`setup_solve_ms`** : métrique principale ;
- transfert final de solution, vérification CPU et reconstruction, séparés.

Le rang utilise la médiane de la somme mesurée par répétition, pas une somme de
médianes. Les échantillons d'échauffement et chaque répétition mesurée sont
conservés. Un résidu estimé par GMRES ne suffit pas : il faut la convergence
déclarée du solveur et le résidu vrai recalculé sur CPU
`||S*x-b|| <= rtol*||b||`, pour chaque répétition, ainsi que des valeurs finies.

Les petits problèmes utilisent une référence CPU directe, avec comparaison de
trace. L'assemblage CUDA est comparé à l'assemblage NumPy sur les cas sous la
limite `--reference-max-dofs` (20 000 par défaut). À grande taille, on évite la
factorisation directe : les polynômes doivent être reproduits à 2e-8 en L2 ; les
cas trigonométriques doivent améliorer d'au moins 5 % l'erreur du maillage de
sélection (avec plancher 1e-10). Ce dernier critère est un contrôle de raffinement,
pas une certification de l'ordre asymptotique. Un vecteur de sonde déterministe
vérifie aussi le matvec GPU contre le matvec CPU du même système.

Une configuration ne peut être classée que si **toutes** ses répétitions prévues
sont présentes et validées. Les échecs de convergence restent des résultats
scientifiques utiles et sont exclus du classement. `completion.json` distingue
une campagne exécutée d'une campagne incomplète ; « completed » n'affirme pas que
tous les chemins ont convergé. `coverage.json` liste séparément les familles
sans candidat convergé et les chemins validés.

## Profilage CUDA et confirmation finale

Les profils sont des workers séparés, après les mesures principales :

- matvec complet, collecte des voisins et produit dense lorsque séparables ;
- BJ complet ;
- ASM : restriction, opération locale, prolongation et total ;
- préconditionneur polynomial ou combiné complet ;
- instrumentation d'une résolution GMRES par `CuPyGMRESProfiler`, donnant les
  appels, durées GPU, synchronisations hôte et opérations du petit système CPU.

Les composants utilisent des événements CUDA. Les zéros sur certaines
sous-étapes signalent une fusion, pas une absence de travail. L'application
`gpu_solve` inclut ses allocations et ses factorisations répétées. Le temps du
GMRES instrumenté est perturbé et ne sert jamais au classement.

Avec `--profile-scope paths`, chaque chemin convergé reçoit son profil à la
meilleure configuration connue sur ce problème ; `families` limite aux gagnants
par famille, `none` désactive le profilage. Il n'y a pas d'appel automatique à
Nsight Compute/Systems dans cette campagne.

Les confirmations `pipeline` utilisent la fonction publique
`solve_advection_diffusion_reaction`, donc comprennent réellement le nouvel
assemblage, les transferts, la résolution et la reconstruction pour chaque
répétition. Le temps est mesuré, pas estimé par addition de médianes. La création
de l'espace DG et l'intégration finale de l'erreur L2 sont hors de cette mesure.

Les compteurs mémoire CuPy enregistrés sont des instantanés du pool après la
résolution, **pas des mesures du pic de VRAM**. Les empreintes d'Arnoldi sont
également consignées.

## Résultats produits

| Fichier | Usage |
|---|---|
| `manifest.json` | Grille, périmètre, tailles, ressources estimées, empreinte des sources |
| `environment.json` | Carte, versions, pilote, paramètres de threading |
| `candidates.csv` | Tous les candidats, y compris ceux qui échouent |
| `best_overall.csv` | Meilleure configuration convergée par cas/taille/ordre |
| `best_per_family.csv` | Meilleure configuration par famille |
| `best_per_path.csv` | Meilleure configuration par chemin d'implémentation |
| `end_to_end.csv` | Mesures réelles finales du calcul complet |
| `profiles.csv` | Index des profils CUDA ; détails dans les JSON des jobs |
| `failures.csv`, `issues.json`, `coverage.json` | Échecs, étapes incomplètes, couverture |
| `report.md`, `completion.json` | Bilan lisible et état de fin |
| `jobs/*.json` | Mesures brutes, résidus, historiques GMRES, détails des profils |
| `specs/*.json`, `logs/*.log` | Spécification reproductible et journal de chaque worker |
| `cache/*/*.npy` | Systèmes partagés, inverses et données de reconstruction |

Les caches sont de simples tableaux NumPy non compressés, chargés en lecture
seule par memory mapping. Aucun pickle n'est utilisé. Le coût d'I/O des caches
est enregistré dans le temps global du worker mais exclu de la mesure du
solveur. Après une interruption, la campagne réutilise les résultats terminés.

## Vérifications effectuées dans cet environnement

Aucun GPU NVIDIA n'est disponible ici. Les nombres de dofs et la grille sont
vérifiés sans GPU. Les tests CPU contrôlent la couverture, la suppression des
combinaisons invalides, le rejet des résultats partiels, le classement, les
timeouts, l'assemblage, la résolution et les confirmations de bout en bout.
La validation CPU de l'orchestrateur est explicitement identifiée
`cpu_reference_check` et n'est pas présentée comme une mesure CUDA.
Les tests GPU inclus devront être exécutés sur la machine équipée.

Le mode de vérification CPU est reproductible avec :

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/run_adv_diff_rea_performance.py \
  --level smoke --cpu-check --warmup 1 --repeats 2 \
  --assembly-warmup 0 --assembly-repeats 2 \
  --component-warmup 1 --component-repeats 3 \
  --output results/adr_performance_cpu_check
```

La vérification livrée dans `results/adr_performance_cpu_verified/` a terminé
avec **27 tests réussis et 10 tests CUDA ignorés**, dix configurations CPU
validées, dix profils CPU et dix confirmations de bout en bout. La reprise a
réutilisé les résultats terminés avec la même empreinte de code. Le manifeste
GPU complet, produit sans exécution CUDA, est dans
`results/adr_performance_full_plan/manifest.json`.

Un exemple de soumission SLURM est fourni dans
`scripts/submit_adv_diff_rea_performance.sbatch`. Soumettre depuis la racine du
projet, avec l'environnement Python/CUDA déjà activé, et ajouter les options de
partition et de compte propres au centre. Les directives par défaut demandent
un GPU, 48 Gio de RAM et 24 h ; la durée effective de la campagne doit être
observée et la reprise utilisée si l'allocation se termine avant la fin.
