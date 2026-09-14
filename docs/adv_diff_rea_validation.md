# Advection–diffusion–réaction HDG : assemblage, validation et chronométrage

## Périmètre et état de validation

Ajout du problème scalaire stationnaire conservatif, sur les maillages triangulaires
2D du projet :

\[
\nabla\cdot(\mathbf q+\boldsymbol\beta u)+ru=f,
\qquad \mathbf q=-K\nabla u.
\]

Advection et convection désignent ici le même transport. Pour résoudre la forme
non conservative `-div(K grad u) + beta.grad(u) + c*u = f`, fournir
`reaction = c - div(beta)`. La vitesse doit être continue à travers les faces ;
chaque composante accepte une constante ou une fonction `(x,y)`. K peut être un
scalaire strictement positif, un tenseur SPD constant ou des composantes variables
au format déjà accepté par `diff_rea`. Les vérifications de positivité sont faites
aux points de quadrature. La frontière entière porte une condition de Dirichlet,
imposée par élimination exacte des traces. Pas encore de Neumann, de périodicité,
de transitoire ni de diffusion nulle dans cette nouvelle API.

La campagne CPU comporte 30 configurations (5 problèmes, 3 maillages, 2 ordres),
avec 1 échauffement puis 3 répétitions et un profilage séparé par configuration.
Toutes ont réussi. Ces résultats ne valident pas l'exécution CUDA : les tests
CUDA inclus doivent encore passer sur une machine NVIDIA.

## Formulation et condensation

Le flux normal numérique est

\[
\widehat{F}_n=\mathbf q_h\cdot\mathbf n+
(\boldsymbol\beta\cdot\mathbf n)\widehat u_h+
\tau(u_h-\widehat u_h),\qquad
\tau=\tau_d+\max(\boldsymbol\beta\cdot\mathbf n,0),\quad \tau_d>0.
\]

Les deux équations locales sont

\[
(K^{-1}\mathbf q_h,\mathbf v)_T-(u_h,\nabla\cdot\mathbf v)_T
+\langle\widehat u_h,\mathbf v\cdot\mathbf n\rangle_{\partial T}=0,
\]
\[
-(\mathbf q_h+\boldsymbol\beta u_h,\nabla w)_T+
\langle\widehat F_n,w\rangle_{\partial T}+(ru_h,w)_T=(f,w)_T.
\]

Avec `x=[u,qx,qy]`, le code construit `A x = f_local + B uhat` et
`flux_moments = C x - D uhat`. Les lignes de flux constitutif sont multipliées
par -1, conformément aux conventions de `diff_rea`. Le système global assemblé
s'écrit `sum(D - C A^{-1} B) uhat = sum(C A^{-1} f_local)` sur les faces intérieures.
L'orientation des colonnes de B et des lignes de C/D est prise en compte avant
l'assemblage global. La continuité du flux numérique est donc imposée au sens
faible pour toutes les fonctions tests de trace.

| Objet | Forme par lot d'éléments |
|---|---|
| A, inverse locale | `(NE, 3P, 3P)` |
| B | `(NE, 3P, 3F)` |
| C | `(NE, 3, F, 3P)` |
| D | `(NE, 3, F, F)` |
| Blocs condensés élémentaires | `(NE, 3, 3, F, F)` |

`P=(p+1)(p+2)/2`, `F=p+1`. Le stockage global reprend le format face-dense
existant. Le solveur CPU direct construit une CSR sans matrice globale dense ;
GMRES CPU applique directement les blocs face-dense.

## Chemin CUDA : hybride, à valider

`assembly_backend='cupy'` réalise l'inversion locale par lots, les produits de
condensation et l'assemblage global par le noyau CUDA existant. Les intégrales
(coefficient, géométrie, quadrature) sont préparées sur CPU. Les résultats sont
rapatriés pour l'assemblage du second membre, l'élimination et la reconstruction.
Le système réduit est transféré de nouveau pour `solver='gpu'`.

GMRES et ses préconditionneurs réutilisent le constructeur face-dense existant :
none, polynomial, Block-Jacobi, BJ-polynomial, ASM et ASM-polynomial. Ce constructeur
est indépendant du modèle PDE, malgré son nom historique diffusion–réaction.
La campagne utilise CGS2, float64 et désactive l'autotuning pour comparer des choix
fixes. L'API accepte les options GPU existantes. Aucun gain GPU n'est annoncé ici.

L'assemblage conserve les inverses mixtes et plusieurs intermédiaires. Il s'agit
d'une première version de référence accélérable, pas d'un assemblage optimisé pour
les très grands p. À p=6, 256×256 cellules rectangles, **une seule** matrice locale
par lot `(131072,84,84)` occupe environ 6,9 Gio en float64 ; les autres tableaux
s'ajoutent. Commencer par les petits cas avant toute campagne sur une V100 32 Gio.

## Exécution

Depuis la racine extraite du projet, utiliser l'environnement Python habituel du
projet (`numpy`, `scipy`, `numba`, `pytest`, et CuPy compatible avec la machine pour
CUDA). Ne pas remplacer l'environnement CUDA du mésocentre pour ces ajouts.

```bash
PYTHONPATH=. python -m pytest -q tests/test_adv_diff_rea.py

OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/run_adv_diff_rea_campaign.py \
  --meshes 4 8 16 --orders 1 2 --repeats 3 --profile \
  --output results/adr_cpu
```

Première vérification CUDA, avec comparaison à la référence CPU directe :

```bash
PYTHONPATH=. python scripts/run_adv_diff_rea_campaign.py \
  --cases quadratic variable_velocity trigonometric \
  --meshes 4 8 --orders 2 --assembly-backend cupy --solver gpu \
  --preconditioner asm_poly --polynomial-degree 6 --restart 75 \
  --rtol 1e-12 --warmup 2 --repeats 5 --validate-gpu \
  --output results/adr_gpu_smoke
```

Après réussite, chronométrage sur un maillage plus grand :

```bash
PYTHONPATH=. python scripts/run_adv_diff_rea_campaign.py \
  --cases trigonometric advection_dominated anisotropic \
  --meshes 32 64 --orders 1 2 3 --assembly-backend cupy --solver gpu \
  --preconditioner asm_poly --polynomial-degree 18 --restart 75 \
  --rtol 1e-11 --warmup 2 --repeats 5 --output results/adr_gpu_timing
```

Pour comparer les préconditionneurs, répéter cette commande en changeant
`--preconditioner` et le répertoire de sortie. La convergence d'ASM-polynomial sur
un problème fortement advectif doit être mesurée ; les paramètres optimaux du
problème de diffusion ne sont pas nécessairement transposables.

## Mesures et profilage

- `environment.json` : arguments, versions et GPU si disponible.
- `runs.jsonl` : temps bruts de chaque répétition mesurée, écrits immédiatement.
- `summary.csv` / `summary.json` : médianes, minimums, maximums, erreurs L2,
  résidus vrais, itérations et taux de convergence observés entre maillages.
- `*.prof` / `*_profile.txt` avec `--profile` : profil CPU séparé, hors mesures.

Toutes les durées sont en secondes. Les phases CUDA sont synchronisées avant
l'arrêt du chronomètre. Ces valeurs sont des **temps muraux**, incluant lancement
et synchronisation, et pour `global_blocks` la construction des tables de
contribution. Ce ne sont pas des temps CUDA-events de noyaux seuls.
`host_preparation`, `host_to_device`, `local_inverse`, `condensation`,
`global_blocks`, `device_to_host`, `rhs_and_boundary`, `assembly_total`,
`solver_setup`, `solve`, `residual_and_reconstruction` et `total` sont séparés.
Le coût maillage/espace est dans `space_setup`, hors `total`.
`setup_plus_solve_median` est la médiane de la somme par répétition ; pour le
solveur direct, `solve` inclut la factorisation creuse. Les warmups sont exclus.
Les sous-phases peuvent ne pas sommer exactement au total (overheads Python).

`--profile` mesure les appels CPU, y compris l'attente CUDA éventuelle. Pour une
chronologie des noyaux et transferts, lancer une **exécution séparée** sous Nsight
Systems si disponible, par exemple :

```bash
PYTHONPATH=. nsys profile --trace=cuda,nvtx,osrt -o adr_cuda \
  python scripts/run_adv_diff_rea_campaign.py \
  --cases trigonometric --meshes 32 --orders 2 \
  --assembly-backend cupy --solver gpu --repeats 1 --output results/adr_nsys
```

Aucune exécution Nsight n'a été faite ici. Ne pas mélanger ses temps avec les
mesures ordinaires. Les tolérances concernent le résidu linéaire vrai du système
réduit, pas l'erreur de discrétisation. Un échec de solveur, un désaccord CPU/GPU,
ou une erreur non finie est enregistré et produit un code de sortie non nul.
Les solutions polynomiales p≥2 sont vérifiées à `2e-8` dans la campagne, avec un
seuil plus strict dans les tests. Les cas trigonométriques doivent présenter une
erreur décroissante lors du raffinement ; les tests imposent aussi un taux minimal.
Le cas dominé par l'advection est lisse : il ne constitue pas une validation de
couches limites sous-résolues ni une preuve d'absence d'oscillations.

## Fichiers ajoutés

- `hdgfem/solvers/adv_diff_rea.py` : formulation et API CPU/CUDA hybride.
- `scripts/adv_diff_rea_cases.py` : cinq solutions manufacturées.
- `scripts/run_adv_diff_rea_campaign.py` : validation, chronométrage, profilage.
- `tests/test_adv_diff_rea.py` : exactitude, flux, limite sans advection,
  convergence, GMRES CPU et tests CUDA disponibles sur une machine équipée.
- Ce document et `results/adr_cpu_validation/` : guide et résultats réels CPU.

Les fichiers préexistants du projet n'ont pas été modifiés.

## Résultats CPU mesurés ici

Machine et versions : voir `environment.json`. Les temps ci-dessous ne sont pas
transposables au T600/P100/V100. Maillage 16×16, médiane de 3 répétitions.

| Cas | p | Erreur L2 | Taux L2 8→16 | Assemblage (ms) | Résolution directe (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|
| quadratic | 1 | 4.694e-03 | 2.00 | 21.50 | 9.79 | 34.19 |
| quadratic | 2 | 2.117e-14 | — | 34.51 | 16.22 | 56.50 |
| variable_velocity | 1 | 4.710e-03 | 2.01 | 20.47 | 5.27 | 27.71 |
| variable_velocity | 2 | 2.775e-14 | — | 34.90 | 15.90 | 54.77 |
| trigonometric | 1 | 1.871e-02 | 1.93 | 21.07 | 5.66 | 28.92 |
| trigonometric | 2 | 9.791e-04 | 2.95 | 32.48 | 15.58 | 51.00 |
| advection_dominated | 1 | 1.492e-02 | 2.11 | 20.88 | 4.53 | 29.21 |
| advection_dominated | 2 | 1.013e-03 | 2.83 | 36.22 | 17.91 | 58.04 |
| anisotropic | 1 | 1.332e-02 | 1.99 | 24.20 | 7.78 | 32.03 |
| anisotropic | 2 | 7.442e-04 | 2.97 | 42.03 | 16.75 | 62.46 |

## Nouvelle campagne comparative complète

La campagne `scripts/run_adv_diff_rea_performance.py` ajoute la comparaison des
chemins, le réglage des paramètres, les profils par événements CUDA, les tailles
proches d'un million de dofs, la reprise et les confirmations de bout en bout.
Voir **`docs/adv_diff_rea_performance.md`** pour les modes `smoke`, `full` et
`exhaustive`, ainsi que leurs périmètres exacts.
