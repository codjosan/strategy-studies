# Roadmap

This roadmap is based on planned project work. Estimates are approximate and
include implementation, tests, and documentation.

| Work item | Planned work | Estimate |
|---|---|---:|
| Add `hlmicro liq-cascade` | Move the liquidation-cascade study behind the package CLI; reuse the shared parsing conventions, expose the current study options, and validate it end to end using synthetic inputs. | 8 hours |
| Run tests automatically on GitHub | Add GitHub Actions for supported Python versions, install the development extras, and run the test suite on pushes and pull requests. | 4 hours |
| Prepare a proper release | Define release/versioning steps, build and verify distributions, document installation and release notes, and publish a tagged release. | 4 hours |
| Add more exchanges | Implement at least one additional venue adapter, normalize its books/trades into existing schemas, and add synthetic fixtures and adapter tests. | 8 hours |

Estimated planned work: **24 hours**.
