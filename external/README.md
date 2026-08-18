# External dependencies

Dependencies in this directory retain their own copyright notices and license
files. They are pinned as Git submodules so their source and modification
history remain distinct from ROSplat-owned code.

## Spirula Studio

- Upstream: <https://github.com/harry7557558/spirula-studio>
- Location: `external/spirula-studio`
- License: GNU General Public License v3.0 (`external/spirula-studio/LICENSE`)
- Purpose: portable Gaussian-splat projection, sorting, and rasterization on
  Vulkan.

ROSplat is also distributed under GPL-3.0, so the licenses are compatible.
The submodule is nevertheless kept unmodified. ROSplat-specific integration
belongs in `native/` and links against the pinned upstream targets.

Clone with external sources initialized:

```bash
git clone --recurse-submodules https://github.com/shadygm/ROSplat.git
```

For an existing checkout:

```bash
git submodule update --init --recursive
```
