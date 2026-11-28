# Kelvin lattice site data

`inc/damn/amr/kelvin/site_data.hpp` holds values on a Kelvin lattice's sites
([sites.md](sites.md)) as damn `BoxData`: whatever a model keeps per site (a scalar, a value per
face, any trivially copyable `T`), ghosts included, moved by damn like any field.

| Type / function | What |
|---|---|
| `SiteArray<T, dim>` | one sublattice (corner or body) of a patch: a `T` per cell over the patch's cells grown by `ghost_width`, in C order, so `box_span` lets copies go by rows |
| `SiteField<T, dim>` | both sublattices |
| `for_each_site_resource(name, field, fn)` | `fn(name + ".corner", field.corner)` and `fn(name + ".body", field.body)`: for a `PatchData`'s `for_each_resource` |
| `site_names({names})` | every name's two resources, e.g. for `copy(fill_ghosts, neighbours, names, ld)` |

A site absent from the domain (`KeyBox::has` false, at a bounded edge) still has a slot; models
leave it be.
