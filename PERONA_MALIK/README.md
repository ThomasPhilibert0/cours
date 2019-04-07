# Pour l'évaluation

-   **Prendre 10 min à chaque fin de séance pour écrire ce qu'on a fait dans le
    fichier [./Rapports.md](./Rapports.md)**
-   le rapport terminal doit être rédiger en Latex dans le répertoire [./rapports](rapports)

# Filtrages d'images par l'équation de la chaleur

## Objectif final, très ambitieux.

On cherche à implémenter des filtres linéaires et non linéaires sur des images,
suivant ce qui est proposé au chapitre 2 (pages 11 à 17) du document
[f./refs/cours-JFA.pdf](refs/cours-JFA.pdf).

Pour cela, on devra donc résoudre des équations du type \(\partial_t u =
div(D(u)\nabla u)\) dans \(\Omega=(0,1)\times(0,1)\) avec une donnée initiale
\(u(0)\) associée directement à une image, et des conditions aux limites à
déterminer.

On cherchera a mettre en oeuvre ces méthodes sur des images variées, et à
discuter les résultats.

## Première partie : recherche de resources, documentation

1.  Trouver et synthétiser de la documentation sur le type d'équation dont il
    s'agit: l'équation de la chaleur linéaire, à coefficients constant, avec
    conditions aux limites de Neuman homogènes. Comment fait-on le lien entre une
    image et les fonctions que l'on manipule ici ?
2.  Trouver de la documentation sur les méthodes de différences finies pour cette
    équation.
3.  Doit-on résoudre un système linéaires ? Si oui décrire le système linéaire à
    résoudre. Quelles sont les méthodes numériques adaptées à la résolution de ce
    système linéaire ?
4.  À quelle(s) modification(s) le filtrage de Perrona-Malik correspond-il ?
    Comment la méthode numérique et la résolution pratique du système se
    généralise-t-elle à cette nouvelle situation ?

## Deuxième partie : programmation dans un cas simplifié

On se place pour cette partie dans le cas de l'équation en dimension \(d=1\),
c'est à dire sur l'intervalle \(\Omega = (0,1)\). L'objectif n'est pas de filtrer
une image, mais de comprendre l'implémentation des méthodes.

1.  Détailler l'architecture du code à implémenter, en essayant d'avoir un code
    modulaire et extensible facilement. Entrées et sorties ? Structuration du
    code en fichiers ? Quels outils de l'ensemble scipy/numpy ?
2.  Déterminer des problèmes modèles, dont les solutions sont connues et
    s'expriment de manière analytique. Ces solutions serviront à tester
    l'implémentation.
3.  Programmer la résolution du problème dans le cas linéaire, puis dans le cas
    non linéaire, et vérifier cette implémentation à l'aide des tests de la
    question 2.

## Troisième partie : le cas général

On va généraliser le code produit dans la partie précédente au cas des images en
2D.

1.  Déterminer comment le code produit précédemment peut se généraliser à la
    dimension 2, dans les cas linéaire, puis non-linaire. Qu'est-ce qui change ?
    Quels sont les fichiers à modifier ?  Recalculer des solutions analytiques
    permettant de faire des tests de vérification. Déterminer les méthodes de
    lecture/écriture des images. Noir et blanc ou couleur ?
2.  Produire le code de résolution du problème en 2D, et le vérifier à l'aide des
    tests calculés en 1.
3.  Trouver des images représentatives pour montrer les propriétés des filtres
    programmés, et illustrer les résultats obtenus.

## refs

Voir le répertoire [./refs](./refs)
