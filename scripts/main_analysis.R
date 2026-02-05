# scripts/main_analysis.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(plm)
  library(lmtest)
  library(sandwich)
  library(car)
})

# -------------------------------------------------------------------
# 0) Chemins 
# -------------------------------------------------------------------
data_file <- "data/FinalDataset.xlsx"

# -------------------------------------------------------------------
# 1) Import
FinalDataset <- read_excel(data_file)

# Nettoyage noms de colonnes (évite espaces/accents problématiques)
names(FinalDataset) <- names(FinalDataset) |>
  str_replace_all("\\s+", " ") |>
  str_trim()

# Vérifications minimales
required_cols <- c("Pays", "Nomdesvariables")
missing_cols <- setdiff(required_cols, names(FinalDataset))
if (length(missing_cols) > 0) {
  stop(
    "Colonnes manquantes dans FinalDataset.xlsx: ",
    paste(missing_cols, collapse = ", "),
    "\nAttendu: Pays, Nomdesvariables + colonnes années."
  )
}

# -------------------------------------------------------------------
# 2) Passage au format long puis panel "large" (une ligne = Pays-Année)
# -------------------------------------------------------------------
FinalLong <- FinalDataset |>
  pivot_longer(
    cols = -c(Pays, Nomdesvariables),
    names_to = "annee",
    values_to = "valeur"
  ) |>
  mutate(
    annee = suppressWarnings(as.integer(annee))
  ) |>
  filter(!is.na(annee))

FinalPanel <- FinalLong |>
  pivot_wider(
    names_from = Nomdesvariables,
    values_from = valeur
  ) |>
  drop_na()

# -------------------------------------------------------------------
# 3) Standardisation des noms de variables
# -------------------------------------------------------------------
clean_varnames <- function(x) {
  x |>
    str_to_lower() |>
    str_replace_all("’|'", "") |>
    str_replace_all("[^a-z0-9]+", "_") |>
    str_replace_all("^_|_$", "")
}

names(FinalPanel) <- clean_varnames(names(FinalPanel))

rename_if_present <- function(df, pattern, newname) {
  if (newname %in% names(df)) return(df)
  idx <- which(str_detect(names(df), pattern))
  if (length(idx) == 1) {
    names(df)[idx] <- newname
  }
  df
}

FinalPanel <- FinalPanel |>
  rename_if_present("dette.*publique|public.*debt|debt", "dettepublique") |>
  rename_if_present("croissance.*pib|gdp.*growth|growth", "croissancepib") |>
  rename_if_present("inflation|cpi", "inflation") |>
  rename_if_present("depenses.*publiques|government.*expenditure|expenditure", "depensespubliques") |>
  rename_if_present("paiements.*interets|interest.*payments|interest", "paiementsinterets") |>
  rename_if_present("chomage|unemployment", "chomage") |>
  rename_if_present("recettes.*publiques|government.*revenue|revenue", "recettespubliques") |>
  rename_if_present("epargne.*interieure|gross.*saving|saving", "epargneinterieure") |>
  rename_if_present("solde.*courant|current.*account|account", "soldecourant") |>
  rename_if_present("population.*65|age_65|old", "population65plus")

# Vérification des variables nécessaires au modèle :contentReference[oaicite:2]{index=2}
model_vars <- c(
  "pays", "annee",
  "dettepublique", "croissancepib", "inflation",
  "depensespubliques", "paiementsinterets", "chomage",
  "recettespubliques", "epargneinterieure", "soldecourant",
  "population65plus"
)

missing_model_vars <- setdiff(model_vars, names(FinalPanel))
if (length(missing_model_vars) > 0) {
  stop(
    "Variables manquantes pour estimer les modèles: ",
    paste(missing_model_vars, collapse = ", "),
    "\nAstuce: affiche names(FinalPanel) et ajuste le mapping rename_if_present()."
  )
}

# -------------------------------------------------------------------
# 4) Panel pdata.frame
# -------------------------------------------------------------------
pdata <- pdata.frame(FinalPanel, index = c("pays", "annee"))

# Formule commune
f <- dettepublique ~ croissancepib + inflation + depensespubliques + paiementsinterets +
  chomage + recettespubliques + epargneinterieure + soldecourant + population65plus

# -------------------------------------------------------------------
# 5) Estimations plm : Poolé / Effets fixes / Effets aléatoires :contentReference[oaicite:3]{index=3}
# -------------------------------------------------------------------
pooling_model <- plm(f, data = pdata, model = "pooling")
fixed_model   <- plm(f, data = pdata, model = "within")
random_model  <- plm(f, data = pdata, model = "random")

print(summary(pooling_model))
print(summary(fixed_model))
print(summary(random_model))

# -------------------------------------------------------------------
# 6) Test de Hausman (FE vs RE) :contentReference[oaicite:4]{index=4}
# -------------------------------------------------------------------
hausman_test <- phtest(fixed_model, random_model)
print(hausman_test)

# -------------------------------------------------------------------
# 7) Erreurs robustes (White) clusterisées par pays
# -------------------------------------------------------------------
robust_se <- function(model) {
  coeftest(model, vcov = vcovHC(model, type = "HC1", cluster = "group"))
}

cat("\n--- Coefficients (SE robustes cluster pays) : Pooling ---\n")
print(robust_se(pooling_model))

cat("\n--- Coefficients (SE robustes cluster pays) : Fixed Effects ---\n")
print(robust_se(fixed_model))

cat("\n--- Coefficients (SE robustes cluster pays) : Random Effects ---\n")
print(robust_se(random_model))

# -------------------------------------------------------------------
# 8) Diagnostics panel utiles
# Hétéroscédasticité (Breusch-Pagan)
# Autocorrélation (pbgtest) : panel serial correlation
# -------------------------------------------------------------------
cat("\n--- Hétéroscédasticité (BP test) : pooling ---\n")
print(bptest(pooling_model))

cat("\n--- Autocorrélation (Baltagi–Wu / BG panel) sur FE ---\n")
print(pbgtest(fixed_model))

# -------------------------------------------------------------------
# 9) Multicolinéarité (VIF) sur OLS classique (approx) :contentReference[oaicite:5]{index=5}
# -------------------------------------------------------------------
ols_model <- lm(
  dettepublique ~ croissancepib + inflation + depensespubliques + paiementsinterets +
    chomage + recettespubliques + epargneinterieure + soldecourant + population65plus,
  data = FinalPanel
)

cat("\n--- VIF (OLS) ---\n")
print(vif(ols_model))

# -------------------------------------------------------------------
# 10) Statistiques descriptives par année + Graphiques :contentReference[oaicite:6]{index=6}
# -------------------------------------------------------------------
summary_stats <- FinalPanel |>
  select(-pays) |>
  group_by(annee) |>
  summarise(
    across(
      .cols = where(is.numeric),
      .fns = list(moy = ~ mean(.x, na.rm = TRUE), sd = ~ sd(.x, na.rm = TRUE))
    ),
    .groups = "drop"
  )

print(head(summary_stats, 10))

debt_by_year <- FinalPanel |>
  group_by(annee) |>
  summarise(moyennedette = mean(dettepublique, na.rm = TRUE), .groups = "drop")

ggplot(debt_by_year, aes(x = annee, y = moyennedette)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Évolution moyenne de la dette publique",
    subtitle = "Pays OCDE",
    x = "Année",
    y = "Dette publique moyenne"
  ) +
  theme_minimal()

chomage_by_year <- FinalPanel |>
  group_by(annee) |>
  summarise(moyennechomage = mean(chomage, na.rm = TRUE), .groups = "drop")

ggplot(chomage_by_year, aes(x = annee, y = moyennechomage)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Évolution moyenne du chômage",
    subtitle = "Pays OCDE",
    x = "Année",
    y = "Taux de chômage moyen"
  ) +
  theme_minimal()

croissance_by_year <- FinalPanel |>
  group_by(annee) |>
  summarise(moyennecroissance = mean(croissancepib, na.rm = TRUE), .groups = "drop")

ggplot(croissance_by_year, aes(x = annee, y = moyennecroissance)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Évolution moyenne de la croissance économique",
    subtitle = "Pays OCDE",
    x = "Année",
    y = "Croissance moyenne"
  ) +
  theme_minimal()
