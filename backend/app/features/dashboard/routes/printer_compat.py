"""
Compatibility printer routes.

These endpoints keep older frontend paths working by mapping them to
current printer service methods.
"""

from flask import jsonify, request

from app.core.middleware.cors import create_options_response
from app.core.paths import IS_CLOUD
from app.features.dashboard.routes import dashboard_bp

# Cloud deployments have no physical printer; the browser's print dialog is the "printer".
BROWSER_PRINTER = 'Browser Print'


def _browser_queue():
    return [{'name': BROWSER_PRINTER, 'status': 'Ready', 'jobs': [], 'job_count': 0, 'is_default': True}]


@dashboard_bp.route('/printer/queues', methods=['GET', 'OPTIONS'])
def printer_queues_compat():
    if request.method == 'OPTIONS':
        return create_options_response()

    if IS_CLOUD:
        return jsonify({'success': True, 'printers': _browser_queue(), 'total': 1})

    try:
        from app.features.print.services.printer_service import PrinterService

        service = PrinterService()
        queues = service.get_printer_queues()
        return jsonify({'success': True, 'printers': queues, 'total': len(queues)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'printers': []}), 500


@dashboard_bp.route('/printer/clear-queue', methods=['POST', 'OPTIONS'])
def printer_clear_queue_compat():
    if request.method == 'OPTIONS':
        return create_options_response()

    if IS_CLOUD:
        return jsonify({'success': True, 'message': 'No print queue to clear (browser printing)', 'cleared': 0})

    try:
        data = request.get_json(silent=True) or {}
        printer_name = data.get('printer_name') or data.get('printer')

        from app.features.print.services.printer_service import PrinterService

        service = PrinterService()

        if printer_name:
            cleared = service.clear_queue(printer_name)
            return jsonify({
                'success': True,
                'message': f'Cleared {cleared} jobs from {printer_name}',
                'cleared': cleared,
            })

        queues = service.get_printer_queues()
        total_cleared = 0
        for queue in queues:
            try:
                total_cleared += service.clear_queue(queue['name'])
            except Exception:
                # Keep clearing remaining printers even if one fails.
                continue

        return jsonify({
            'success': True,
            'message': f'Cleared {total_cleared} jobs from all printers',
            'cleared': total_cleared,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@dashboard_bp.route('/printer/cancel-job', methods=['POST', 'OPTIONS'])
def printer_cancel_job_compat():
    if request.method == 'OPTIONS':
        return create_options_response()

    if IS_CLOUD:
        return jsonify({'success': True, 'message': 'No print jobs to cancel (browser printing)'})

    try:
        data = request.get_json(silent=True) or {}
        printer_name = data.get('printer_name') or data.get('printer')
        job_id = data.get('job_id') or data.get('id')

        if not printer_name or not job_id:
            return jsonify({'success': False, 'error': 'printer_name and job_id are required'}), 400

        from app.features.print.services.printer_service import PrinterService

        service = PrinterService()
        service.cancel_job(str(printer_name), str(job_id))
        return jsonify({'success': True, 'message': 'Job cancelled'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@dashboard_bp.route('/printer/diagnostics', methods=['GET', 'OPTIONS'])
def printer_diagnostics_compat():
    if request.method == 'OPTIONS':
        return create_options_response()

    if IS_CLOUD:
        info = {'mode': 'browser', 'printer': BROWSER_PRINTER, 'spooler': 'not applicable'}
        return jsonify({'success': True, 'diagnostics': info, 'output': info})

    try:
        from app.features.print.services.printer_service import PrinterService

        service = PrinterService()
        diagnostics = service.get_diagnostics()

        return jsonify(
            {
                'success': True,
                'diagnostics': diagnostics,
                'output': diagnostics,
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
